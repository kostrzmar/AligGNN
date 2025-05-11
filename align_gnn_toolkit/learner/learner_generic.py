
from learner import AbstractLearner
from model import GenericModel
from utils import config_const
from utils import executor_utils as utils
import time
import logging
from datetime import datetime
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import pearsonr 
from scipy.stats import spearmanr 
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score
import os
import torch.nn.functional as F
from enum import Enum
import numpy as np
import json
import pandas as pd
from torch_geometric.nn import to_hetero
from sentence_transformers.util import pairwise_cos_sim

class SiameseDistanceMetric(Enum):

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)
    HAMMING_SIM = lambda x, y : torch.mean(torch.tanh(x) * torch.tanh(y), dim=1)


class LearnerGeneric(AbstractLearner):
    
    def __init__(self,
                params=None, 
                mlflow=None
                ) -> None:
        super(LearnerGeneric, self).__init__(params, mlflow)
        if self.params[config_const.CONF_MLFLOW_EXPERIMENT_NAME_GENERATE]:
            if config_const.CONF_EXPERIMENT_CURRENT_TRAIL_NUMBER not in params or params[config_const.CONF_EXPERIMENT_CURRENT_TRAIL_NUMBER] ==1:
                self.params[config_const.CONF_MLFLOW_EXPERIMENT_NAME] = utils.generate_unique_exp_name(self.params) 
        self.experiment_name = self.params[config_const.CONF_MLFLOW_EXPERIMENT_NAME]
        #self.DISPLAY_INFO_EVERY = 0.005 # every 1 / 100 batch
        self.INCLUDE_SOFT_F1 = False    
        self.loss_function_name = self.getParamByName("loss_function.name", "MSE", return_as_array=False)
        self.loss_coefficients = self.getParamByName("loss_function.coefficients", 1.0, return_as_array=False)
        self.verify_test_on_best_val = False
        self.ignore_val = self.getParamByName(config_const.CONF_EXPERIMENT_IGNORE_VALUATION, True)
        self.lock_conv = self.getParamByName(config_const.CONF_MODEL_LOCK_CONV_LAYERS, False)
        self.lock_conv_after_epoch =   self.getParamByName(config_const.CONF_MODEL_LOCK_CONV_LAYERS_AFTER_EPOCH, 0)
        self.store_on_metrics =  self.getParamByName(config_const.CONF_MODEL_STORE_BEST_MODEL_ON_METRIC)

        
    def getLocalParameters(self):
        return {}
    
    def getModel(self):
        to_hetero = False
        if to_hetero:
            model = GenericModel(self.get_model_params(), (-1,-1), (-1,-1), None)        
            data = self.data_holder.train_data_set[0]
            with torch.no_grad():  # Initialize lazy modules.
                out = model(data.x_dict, data.edge_index_dict)
                return to_hetero(self.model, data.metadata(), aggr='sum')
        else:
            if self.embed_one_hot is not None and self.embed_one_hot:
                new_node_feature_size = 0
                new_edge_feature_size = 0
                nodes, edges = self.get_graph_builder().getGraphOffsets()
                for node in nodes:
                    if node[0]:
                        new_node_feature_size+=self.one_hot_embed_size
                    else:
                        new_node_feature_size+=node[2]
                for edge in edges:
                    if edge[0]:
                        new_edge_feature_size+=self.one_hot_embed_size
                    else:
                        new_edge_feature_size+=edge[2]
                return GenericModel(self.get_model_params(), new_node_feature_size, new_edge_feature_size, None, embed_one_hot=(nodes, edges)) 
            else:
                return GenericModel(self.get_model_params(), self.get_feature_size(), self.get_edge_feature_size(), None)        

    def loadModel(self, path_to_model, model_type):
        return utils.load_model(self.getModel(),path_to_model,model_type)

    def convert_logits(self, y_pred_logits):
        y_pred = y_pred_logits
        if self.logits_converter == 'sigmoid':
            y_pred = torch.sigmoid(y_pred_logits)
        return y_pred
    
    def get_storage(self,items):
        storage = {}
        for item in items:
            storage[item] = {}
            storage[item]['min'] = None
            storage[item]['max'] = None
            storage[item]['current'] = None
        return storage  

    def add_to_storage(self, storage, type, item):
        storage[type]['current'] = item
        min = storage[type]['min']
        max = storage[type]['max']
        if not max or item > max:
            storage[type]['max'] = item
        if not min or item < min:
            storage[type]['min'] = item
            
    def get_from_storage(self, storage, type):
        return f'{storage[type]["current"]:.04f}|{storage[type]["min"]:.03f}|{storage[type]["max"]:.03f}' 


    def display_progress_info(self,index,dataset_size, loss, loss_all, y_pred, y_gold, storage=None, phase=None):
        if storage:
            self.add_to_storage(storage, "mse", mean_squared_error(y_pred, y_gold))
            self.add_to_storage(storage,"p", pearsonr(y_pred, y_gold)[0])
            self.add_to_storage(storage, "s", spearmanr(y_pred, y_gold)[0])
            self.add_to_storage(storage, "loss", loss)
            print(f'{phase}_I:{(index/dataset_size)*100:.02f}% L:{self.get_from_storage(storage, "loss")} LA:{loss_all:.04f} M:{self.get_from_storage(storage, "mse")} P:{self.get_from_storage(storage, "p")} S:{self.get_from_storage(storage, "s")} ', end='\r')   
        else:    
            print(f'Train Index: {(index/dataset_size)*100:.02f}% Loss: {loss:.04f} LossAll: {loss_all:.04f}', end='\r')

    def do_predict(self, model, embedding_as_numpy=True, data_loader=None):
        model.eval()
        predict_y = []
        gold_y = []
        emb1 = []
        emb2 = []
        model.to(self.device)
        local_data_loader = self.data_holder.test_data_loader
        if data_loader:
            local_data_loader = data_loader
        for index, batch in enumerate(tqdm(local_data_loader)):
                    
            graph = batch
            graph = graph.to(self.device)
            gold_y.extend((graph.y).detach().cpu())
            with torch.no_grad():
                y_pred_logits,emb_src,emb_trg, l, e =model(graph)
                y_pred = self.convert_logits(y_pred_logits)
                predict_y.extend(y_pred.detach().cpu())
                emb1.extend(emb_src.detach().cpu())
                emb2.extend(emb_trg.detach().cpu())
        if embedding_as_numpy:
            embeddings1 = np.asarray([emb.numpy() for emb in emb1])
            embeddings2 = np.asarray([emb.numpy() for emb in emb2])
        else:
            embeddings1 = emb1
            embeddings2 = emb2
        return predict_y, gold_y, embeddings1, embeddings2

            
    def do_training(self, epoch,  data_loader, is_training=True, data_loader_type=None):
        if is_training:  
            self.model.train()
        else:
            self.model.eval()
            g_y_all = []
            p_y_all = []
        total_loss = 0
        local_model_metrics = {}
        storage = self.get_storage(['mse', 'p', 's', 'loss', 'loss_all'])   
        dataset_size  = len(data_loader)
        total_dataset_size = len(data_loader.dataset)
        for index, data in enumerate(data_loader):
            data = data.to(self.device)
            if is_training:  
                self.optimizer.zero_grad()
                y_pred_logits, emb_s,emb_t, l, e = self.model(data)
            else:
                with torch.no_grad():
                    y_pred_logits, emb_s,emb_t, l, e = self.model(data)
                    g_y=  (data.y).float().detach().cpu().numpy()
                    y_pred = self.convert_logits(y_pred_logits)
                    p_y= y_pred.float().detach().cpu().numpy()
                    g_y_all.extend(g_y)
                    p_y_all.extend(p_y)
            loss = self.do_loss(y_pred_logits, data.y, emb_s, emb_t, self.loss_function_name, self.loss_coefficients)
            loss = loss + l + e  
            
            if index % self.show_info_by ==0:
                if is_training:  
                    y_pred = self.convert_logits(y_pred_logits)
                    self.display_progress_info(index,dataset_size, loss, total_loss, y_pred.detach().cpu(), data.y.detach().cpu(), storage, 'T')
                else:
                    self.display_progress_info(index,dataset_size, loss, total_loss, y_pred.detach().cpu(), data.y.detach().cpu(), storage, 'V')

            if is_training:  
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item() * data.num_graphs   
        if not is_training:

            r2, mqe, mae, rmqe, pear, spear, roc_auc= 0,0,0,0, 0,0,0
            if "R2_score" in self.store_on_metrics:
                r2 = r2_score(g_y_all , p_y_all)
            if "MQE"  in self.store_on_metrics:
                mqe=mean_squared_error(g_y_all , p_y_all)
            if "MAE"  in self.store_on_metrics:
                mae=mean_absolute_error(g_y_all , p_y_all)  
            if "RMQE"  in self.store_on_metrics: 
                rmqe= np.sqrt(mean_squared_error(g_y_all , p_y_all)) 
            if "pearsonr" in self.store_on_metrics: 
                pear=pearsonr(g_y_all , p_y_all)[0]   
            if "SpearmanR" in self.store_on_metrics: 
                spear=spearmanr(g_y_all , p_y_all)[0]      
            if "roc_auc" in self.store_on_metrics:
                gold_int = self.convert_gold_to_int(g_y_all).numpy() 
                roc_auc=roc_auc_score(gold_int , p_y_all)
                  
            local_model_metrics = self.get_model_metric(
                        type=data_loader_type,
                        loss=total_loss / total_dataset_size,
                        r2=r2,
                        mqe=mqe,
                        mae=mae,
                        rmqe= rmqe,
                        pear=pear,
                        spear=spear,
                        roc_auc=roc_auc, 
                        epoch=epoch
            )
        
        return   total_loss  / total_dataset_size, local_model_metrics    

    def convert_gold_to_int(self, all_gold):
        return torch.Tensor([ 1 if x == 1.0 or x==1 else 0 for x in all_gold])
   
    def get_model_metric(self, type, loss=0, r2=0, mqe=0, mae=0, rmqe=0, pear=0, spear=0, roc_auc=0,epoch=0):
        local_model_metrics = {
                        "type":type,
                        "loss": loss,
                        "R2_score":r2,
                        "MQE":mqe,
                        "MAE":mae,
                        "RMQE" : rmqe,
                        "pearsonr":pear,
                        "SpearmanR":spear,
                        "roc_auc":roc_auc,
                        "epoch": epoch,
                        }
        return local_model_metrics
        

    def _convert_to_tensor(self, a) -> torch.Tensor:
        """
        Converts the input `a` to a PyTorch tensor if it is not already a tensor.

        Args:
            a (Union[list, np.ndarray, Tensor]): The input array or tensor.

        Returns:
            Tensor: The converted tensor.
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)
        return a

    def pairwise_angle_sim(self, x: torch.Tensor, y: torch.Tensor):
        # https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py
        x = self._convert_to_tensor(x)
        y = self._convert_to_tensor(y)

        # modified from https://github.com/SeanLee97/AnglE/blob/main/angle_emb/angle.py
        # chunk both tensors to obtain complex components
        a, b = torch.chunk(x, 2, dim=1)
        c, d = torch.chunk(y, 2, dim=1)

        z = torch.sum(c**2 + d**2, dim=1, keepdim=True)
        re = (a * c + b * d) / z
        im = (b * c - a * d) / z

        dz = torch.sum(a**2 + b**2, dim=1, keepdim=True) ** 0.5
        dw = torch.sum(c**2 + d**2, dim=1, keepdim=True) ** 0.5
        re /= dz / dw
        im /= dz / dw

        norm_angle = torch.sum(torch.concat((re, im), dim=1), dim=1)
        return torch.abs(norm_angle)

        

    def calculate_loss(self, pred_logits, gold, emb_s=None, emb_t=None, loss_type=None):
        loss = 0.0
        if loss_type == "BCE":
            pos_weight = torch.FloatTensor([50]) # was 4 or 50
            #loss_raw = F.binary_cross_entropy(pred, gold,reduction="none")
            loss_raw = F.binary_cross_entropy_with_logits(pred_logits, gold,reduction="none")
            weight = torch.ones_like(loss_raw)
            weight[gold==1.] = pos_weight
            loss = ((loss_raw) * weight).mean() #* 20 #+ 1 - soft_f1 # was 20  
            #return ((loss_raw) * weight).sum() #* 20 #+ 1 - soft_f1 # was 20   
        elif loss_type == "COS":
            gold = gold*2-1
            loss =  F.cosine_embedding_loss(emb_s, emb_t, gold)
        elif loss_type == "COS_MSE":
            score = torch.cosine_similarity(emb_s, emb_t)
            loss =  F.mse_loss(score, gold) 
        elif loss_type =="CONT":
            distances = SiameseDistanceMetric.COSINE_DISTANCE(emb_s, emb_t)
            losses = 0.5 * (
                gold.float() * distances.pow(2) + (1 - gold).float() * F.relu(1.2 - distances).pow(2)
            )
            loss =  losses.mean() #if self.size_average else losses.sum()
        elif loss_type =="CoSENT":
            #from -> https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/CoSENTLoss.py#L13-L114
            scale: float = 20.0
            scores = pairwise_cos_sim(emb_s, emb_t)
            scores = scores * scale
            # label matrix indicating which pairs are relevant
            labels = gold[:, None] < gold[None, :]
            labels = labels.float()
            # mask out irrelevant pairs so they are negligible after exp()
            scores = scores - (1 - labels) * 1e12
            # append a zero as e^0 = 1
            scores = torch.cat((torch.zeros(1).to(scores.device), scores.view(-1)), dim=0)
            loss = torch.logsumexp(scores, dim=0)
        elif loss_type =="AnglELoss":
            #from -> https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/CoSENTLoss.py#L13-L114
            scale: float = 20.0
            scores = self.pairwise_angle_sim(emb_s, emb_t)
            
            scores = scores * scale
            # label matrix indicating which pairs are relevant
            #labels = gold[:, None] < gold[None, :]
            labels = gold[None, :] < gold[:, None]
            labels = labels.float()
            # mask out irrelevant pairs so they are negligible after exp()
            scores = scores - (1 - labels) * 1e12
            
            # append a zero as e^0 = 1
            scores = torch.cat((torch.zeros(1).to(scores.device), scores.view(-1)), dim=0)
            
            loss = torch.logsumexp(scores, dim=0)
            
        
        elif loss_type =="ONLINE_CONT":
            distance_matrix = SiameseDistanceMetric.COSINE_DISTANCE(emb_s, emb_t)
            negs = distance_matrix[gold == 0]
            poss = distance_matrix[gold == 1]

            # select hard positive and hard negative pairs
            negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
            positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

            positive_loss = positive_pairs.pow(2).sum()
            negative_loss = F.relu(0.5 - negative_pairs).pow(2).sum()
            loss = positive_loss + negative_loss
        elif loss_type =="CO_SENT":
            scores = SiameseDistanceMetric.COSINE_DISTANCE(emb_s, emb_t)
            scores = scores * 20
            scores = scores[:, None] - scores[None, :]
            gold = gold*2-1
            # label matrix indicating which pairs are relevant
            labels = gold[:, None] < gold[None, :]
            labels = labels.float()
            # mask out irrelevant pairs so they are negligible after exp()
            scores = scores - (1 - labels) * 1e12
            # append a zero as e^0 = 1
            scores = torch.cat((torch.zeros(1).to(scores.device), scores.view(-1)), dim=0)
            loss = torch.logsumexp(scores, dim=0)
        elif loss_type =="MSE":
            y_pred = self.convert_logits(pred_logits)
            loss =  F.mse_loss(y_pred, gold)

        elif loss_type =="MNRL_COS":
            scale = 20
            scores = torch.stack([
                F.cosine_similarity(
                    a_i.reshape(1, a_i.shape[0]), emb_t
                ) for a_i in emb_s])
            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)       
            loss = F.cross_entropy(scores*scale, labels)
        elif loss_type =="MNRL_DISTANCE":
            scale = 20
            scores3 = torch.exp(
                -torch.stack(
                    [F.pairwise_distance(a_i.reshape(1, a_i.shape[0]), emb_t, p=2) 
                    for a_i in emb_s]))
            labels3 = torch.tensor(range(len(scores3)), dtype=torch.long, device=scores3.device)  # Example a[i] should mat ch with b[i]
            loss =  F.cross_entropy(scores3*scale, labels3)  
        elif loss_type =="MNRL_DOT":
            scale = 1
            scores2 = dot_score(emb_s, emb_t) 
            labels2 = torch.tensor(range(len(scores2)), dtype=torch.long, device=scores2.device)  # Example a[i] should match with b[i]
            loss =  F.cross_entropy(scores2* scale, labels2)   
        return loss

    def do_loss(self, pred_logits, gold, emb_s=None, emb_t=None, loss_types=['COS'], loss_coefficients=[1.0]):
        soft_f1 = 0
        if self.INCLUDE_SOFT_F1:
            if "BCE"  in loss_types:
                y_pred = torch.sigmoid(pred_logits)
            else:
                y_pred = F.relu(F.cosine_similarity(emb_s, emb_t, dim=-1))
            tp = torch.sum(torch.mul(gold, y_pred))
            fp = torch.sum(torch.mul(1 - gold, y_pred))
            fn = torch.sum(torch.mul(gold, 1 - y_pred))
            soft_f1 = (2 * tp) / (2 * tp + fn + fp + 1e-16)
            soft_f1 = 1 - soft_f1
        final_loss = soft_f1
        for loss_type, loss_coefficient in zip(loss_types, loss_coefficients): 
            loss = self.calculate_loss(pred_logits, gold, emb_s, emb_t, loss_type)
            final_loss+= loss*loss_coefficient

        return final_loss

    
    
    def compare_results(self, best_eval, current_eval, is_lower_better=True):
        if is_lower_better:
            return best_eval is None or (current_eval) < (best_eval)
        else:
            return best_eval is None or (current_eval) > (best_eval)    

    def store_best_results(self, type, metric_type, value, model_performance, local_model_metrics_test=None):
        has_result_improved = False
        is_lower_better = False
        if metric_type == 'best_loss':
            is_lower_better=True
        elif metric_type == "best_mqe":
            is_lower_better=True
        elif metric_type == 'best_SpearmanR':
            is_lower_better=False
        elif metric_type == 'best_pearsonr':
            is_lower_better=False
        elif metric_type =="best_roc_auc":
            is_lower_better=False
            
        
        if self.compare_results(model_performance.get(type, metric_type),value, is_lower_better=is_lower_better):
            utils.save_model_raw(self.model,self.experiment_path, local_model_metrics_test, type+"_"+metric_type)
            model_performance.set(type, metric_type ,value)
            has_result_improved = True
        return has_result_improved  


    def initialize_trail_stats(self, trials_stats):
        if 'test_loss_min' not in trials_stats:
            trials_stats['test_R2_score_mean'] = []
            trials_stats['test_pearsonr_mean'] = []
            trials_stats['test_SpearmanR_mean'] = []
            trials_stats['test_MQE_mean'] = []
            trials_stats['test_MAE_mean'] = []
            trials_stats['test_roc_auc_mean'] = []
            trials_stats['test_loss_min'] = []


    def update_best_per_trail(self, metric, model_metrics, trials_stats, is_lower_better=True):
        should_update = False
        if is_lower_better:
            if metric not in trials_stats or model_metrics[metric] < trials_stats[metric+"_per_trail"]:
                should_update = True
        else:
            if metric not in trials_stats or model_metrics[metric] > trials_stats[metric+"_per_trail"]:
                should_update = True
        if should_update:
            trials_stats[metric+"_per_trail"] =  model_metrics[metric]
            trials_stats[metric+"_per_trail_model"] = self.experiment_path
            trials_stats[metric+"_per_trail_experiment_path"] = model_metrics["experiment_path"]


    def fit(self, tune=None, trials_stats=None):  

        show_process_info =True
        self.initialize_trail_stats(trials_stats)
        self.epoch_number=self.getParamByName("train.epoch_number", 101)
        early_stop_by = self.getParamByName("train.early_stop_by", 10)
        early_stopping_counter = 0
        early_stopping = False
        model_metrics = {}
        model_metrics["experiment"]=self.experiment_name
        model_metrics["experiment_timestamp"]=datetime.now().strftime("%m%d_%H%M")
        model_metrics["nbr_trials"]=  self.getParamByName(config_const.CONF_EXPERIMENT_CURRENT_TRAIL_NUMBER,1)
        model_performance = utils.ModelPerformance()
        
        self.model = self.model.to(self.device) 
        # this zero gradient update is needed to avoid a warning message
        self.optimizer.zero_grad()
        self.optimizer.step()
            
        perf_per_epoch = [] 
        t = time.time()
        model_metrics["per_epoch"]  = {}  
        pbar = tqdm(range(1, self.epoch_number+1))
        for epoch in pbar:
            if epoch % self.show_info_by == 0:
                pbar.disable = False
            else:
                pbar.disable = True               
            model_metrics["per_epoch"][epoch]=[]
            if early_stopping_counter <= early_stop_by:
                if self.lock_conv and epoch > self.lock_conv_after_epoch:
                    logging.info("Lock convolution against training")
                    self.model.lock_convolution()
                    self.lock_conv=False
                train_loss, _ = self.do_training(epoch,  self.data_holder.train_data_loader, is_training=True)                

                if self.store_best_results("train", "best_loss", train_loss, model_performance):
                    self.best_train_loss=train_loss
                    
                if not self.ignore_val:
                    val_loss, local_model_metrics_val = self.do_training(epoch,  self.data_holder.validation_data_loader, is_training=False, data_loader_type="val")
                    model_metrics["per_epoch"][epoch].append(local_model_metrics_val)
                    
                    if self.store_best_results("val", "best_loss", val_loss, model_performance, local_model_metrics_val):
                        self.verify_test_on_best_val = True

                else: 
                    val_loss =-1
                    local_model_metrics_val = self.get_model_metric(type="val")
                    model_metrics["per_epoch"][epoch].append(local_model_metrics_val)
                
                if (not self.ignore_val and self.verify_test_on_best_val) or (self.ignore_val):
                    self.verify_test_on_best_val = False
                    test_loss, local_model_metrics_test = self.do_training(epoch,  self.data_holder.test_data_loader, is_training=False, data_loader_type="test")
                    model_metrics["per_epoch"][epoch].append(local_model_metrics_test)
                
                    if self.store_best_results("test", "best_loss", test_loss, model_performance, local_model_metrics_test):
                        early_stopping_counter = 0
                        self.best_test_loss=test_loss                        
                    else:
                        early_stopping_counter += 1  
                
        
                    if "MQE" in self.store_on_metrics:
                        self.store_best_results("test", "best_mqe", local_model_metrics_test["MQE"], model_performance, local_model_metrics_test)
                    if "SpearmanR" in self.store_on_metrics:
                        self.store_best_results("test", "best_SpearmanR", local_model_metrics_test["SpearmanR"], model_performance, local_model_metrics_test)
                    if "pearsonr" in self.store_on_metrics:
                        self.store_best_results("test", "best_pearsonr", local_model_metrics_test["pearsonr"], model_performance, local_model_metrics_test)
                    if "roc_auc" in self.store_on_metrics:
                        self.store_best_results("test", "best_roc_auc", local_model_metrics_test["roc_auc"], model_performance, local_model_metrics_test)


                    if tune and self.getParamByName("ray.enable", False):
                        tune.report(min_loss=test_loss)                        
                  
                else:                 
                    if epoch ==0:
                        model_metrics["per_epoch"][epoch].append(self.get_model_metric(type="test"))
                    else:
                        model_metrics["per_epoch"][epoch].append(model_metrics["per_epoch"][epoch-1][1])
                    

                if isinstance(self.scheduler._scheduler, ReduceLROnPlateau):
                    self.scheduler.step(train_loss)
                else:
                    self.scheduler.step()
                lr = self.scheduler.optimizer.param_groups[0]['lr']
                                   
                if epoch % self.show_epoch_info_by == 0:
                    out = f'[{self.experiment_name[:15]}], E: {epoch:03d}, LR: {lr:4f}, L: {train_loss:.4f}{utils.show_direction(train_loss,model_performance.get("train", "best_loss"))}/{val_loss:.4f}{utils.show_direction(val_loss,model_performance.get("val", "best_loss"))}/{test_loss:.4f}{utils.show_direction(test_loss,model_performance.get("test", "best_loss"))}'
                    for metric in self.store_on_metrics:
                        out += f', T_{metric[0].upper()} {(local_model_metrics_test[metric]):.4f}|{(model_performance.get("test", "best_"+metric)):.4f} '    
                    print(out)

                utils.store_per_per_epoch(perf_per_epoch, local_model_metrics_test["MQE"], local_model_metrics_val["MQE"], epoch, lr,train_loss, val_loss,test_loss, self.experiment_name, model_metrics, calculate_mean=False)
            else:
                early_stopping = True
                break                       
        
        t = time.time() - t
        train_time = t/60    
        model_metrics["best_train_loss"]=model_performance.get("train", "best_loss")    
        model_metrics["best_val_loss"]=model_performance.get("val", "best_loss")
        model_metrics["best_test_loss"]=model_performance.get("test", "best_loss")
        model_metrics["best_test_mqe"]=model_performance.get("test", "best_mqe")
        model_metrics["best_test_SpearmanR"]=model_performance.get("test", "best_SpearmanR")
        model_metrics["best_test_pearsonr"]=model_performance.get("test", "best_pearsonr")
        model_metrics["best_test_roc_auc"]=model_performance.get("test", "best_roc_auc")        
        
        model_metrics["train_time"]=train_time
        model_metrics["perf_per_epoch"]=perf_per_epoch
        model_metrics["experiment_path"]=self.experiment_name
        pbar.close()
        
        if early_stopping:
            logging.info(f'Early stopping at epoch: {epoch} after {early_stop_by} steps')
        
        def _format(value):
            if value:
                return f'{value:.4f}'
            return 'N/A'
        
        logging.info('Best Train Loss:\t'+_format(model_performance.get("train", "best_loss")))   
        logging.info('Best Validation Loss:\t'+_format(model_performance.get("val", "best_loss")))
        logging.info('Best Test Loss:\t\t'+_format(model_performance.get("test", "best_loss")))
        logging.info('Best Test MQE:\t\t'+_format(model_performance.get("test", "best_mqe")))
        logging.info('Best Test SpearmanR:\t\t'+_format(model_performance.get("test", "best_SpearmanR")))
        logging.info('Best Test Pearsonr:\t\t'+_format(model_performance.get("test", "best_pearsonr"))) 
        logging.info('Best Test Roc_auc:\t\t'+_format(model_performance.get("test", "best_roc_auc")))       
        
        
        out_statistic = {}
        out_statistic['Epoch']=str(epoch)
        out_statistic['Train loss']=str(round(self.best_train_loss,4))
        out_statistic['Test loss']=str(round(self.best_test_loss,4))
        out_statistic['trail_model_path'] = self.experiment_path
        for key in model_metrics:
            item = model_metrics[key]
            if item and  isinstance(item, float):
                out_statistic[key]=str(model_metrics[key])
        with open(os.path.join(self.experiment_path, "results.json"), "w") as wf:
            json.dump(out_statistic, wf, indent=4)    

        self.update_best_per_trail("best_test_mqe", model_metrics, trials_stats, is_lower_better=True)
        self.update_best_per_trail("best_test_SpearmanR", model_metrics, trials_stats, is_lower_better=False)
        self.update_best_per_trail("best_test_pearsonr", model_metrics, trials_stats, is_lower_better=False)
        self.update_best_per_trail("best_test_loss", model_metrics, trials_stats, is_lower_better=True)         
        self.update_best_per_trail("best_test_roc_auc", model_metrics, trials_stats, is_lower_better=False)
                
        metrics = ['mean','std', 'min']
        avg_results = utils.get_final_performance_raw(model_metrics["perf_per_epoch"], 
                {'test_R2_score':metrics, 'test_pearsonr':metrics, 
                'test_SpearmanR':metrics, 'test_MQE':metrics, 'test_MAE':metrics, 
                'loss':metrics, 'test_loss':metrics, "test_roc_auc":metrics
                })
        trials_stats['test_R2_score_mean'].append(avg_results["test_R2_score_mean"])
        trials_stats['test_pearsonr_mean'].append(avg_results['test_pearsonr_mean'])
        trials_stats['test_SpearmanR_mean'].append(avg_results['test_SpearmanR_mean'])
        trials_stats['test_MQE_mean'].append(avg_results['test_MQE_mean'])
        trials_stats['test_MAE_mean'].append(avg_results['test_MAE_mean'])
        trials_stats['test_roc_auc_mean'].append(avg_results['test_roc_auc_mean'])
        trials_stats['test_loss_min'].append(avg_results['test_loss_min'])        

        out_statistic["perf_per_epoch"] = model_metrics["perf_per_epoch"]
        
        return self.best_test_loss, out_statistic




