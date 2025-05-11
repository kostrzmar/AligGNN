from abc import abstractmethod
import logging
import torch
from torch import optim
from data_set import DataSetFactory
from datetime import datetime
import utils.executor_utils as utils_processing
import os
import json
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr 
from scipy.stats import spearmanr 
from utils import config_const
from learner.scheduler_wramupLR import WarmupLR
import sys
import shutil
from utils import executor_utils as utils

class AbstractLearner():
    def __init__(self,
            params=None, 
            mlflow=None
            
            ) -> None:
        self.params = params
        self.mlflow = mlflow
        if self.getLocalParameters():
            self.params.update(self.getLocalParameters())
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data_holder = self.getDataHolder()
        self.embed_one_hot = self.getParamByName(config_const.CONF_MODEL_EMBED_ONE_HOT, False)
        self.one_hot_embed_size = self.getParamByName(config_const.CONF_MODEL_ONE_HOT_EMBEDDING_SIZE)
        self.model = self.getModel()
        self.model = self.model.to(self.device)
        self.optimizer = self.getOptimizer()
        self.scheduler = self.getScheduler()
        self.loss_contrastive = False
        self.loss_function_name = self.getParamByName("loss_function.name", "MSE")
        self.loss_function = self.getLossFunction()
        self.epoch_number = self.getParamByName("train.epoch_number", 101)
        self.early_stop_by = self.getParamByName("train.early_stop_by", 10)
        self.early_stopping_counter = 0        
        self.logits_converter = self.getParamByName(config_const.CONF_TRAIN_LOGITS_CONVERTER)
        self.model_metrics = {}
        self.best_model_metric = {}
        self.best_train_loss=1e+5
        self.best_test_loss=1e+5
        self.show_info_by = self.getParamByName("train.show_info_by", 1)
        self.show_epoch_info_by = self.getParamByName(config_const.CONF_TRAIN_SHOW_EPOCH_INFO_BY, 1)
        self.experiment_path = self.get_experiment_path()
        self.save_experiment_params()
        utils_processing.get_env_info()
        logging.info(f"Number of model's parameters: {self.count_parameters():,}")
        if self.mlflow :
            self.mlflow.log_param("num_params", self.count_parameters())
            for key in self.params.keys():                self.mlflow.log_param(key, self.params[key])
            
            
    
    @abstractmethod
    def getLocalParameters(self):
        pass
    
    @abstractmethod
    def getModel(self):
        pass

    
    def do_training(self, epoch,  data_loader, is_training=True):
        y_true = list()
        y_pred = list()
        total_loss = 0        
        for batch in data_loader:
            batch.to(self.device)  
            out, emb_a, emb_b, _, _ = self.model(batch)  
            if  self.loss_contrastive:
                loss = self.loss_function(emb_a, emb_b, batch.y)
            else:
                loss = self.loss_function(out, batch.y)
            
            if is_training:  
                loss.backward()  
                self.optimizer.step()  
                self.optimizer.zero_grad()
                
            y_true += list(batch.y.float().detach().cpu().numpy())
            y_pred += list(out.float().detach().cpu().numpy())
            total_loss += loss.float().detach().cpu()
            
        self.setModelMetric("R2_score", r2_score(y_true, y_pred))
        self.setModelMetric("pearsonr", pearsonr(y_true, y_pred)[0])
        self.setModelMetric("SpearmanR", spearmanr(y_true, y_pred)[0])
        loss =  np.nanmean(total_loss.data.float())
        return loss, emb_a, emb_b, out, batch.y

    
    
    
    @abstractmethod
    def getDataHolder(self):
        return DataSetFactory.get_data_holder(params=self.params)
    
    @abstractmethod  
    def getOptimizer(self):
        optimizer = self.getParamByName(config_const.CONF_OPTIMIZER_NAME, "Adam")
        if optimizer == "Adam":
            return optim.Adam(self.model.parameters(), 
                              lr = self.getParamByName(config_const.CONF_OPTIMIZER_LEARNING_RATE, 1e-2),
                              weight_decay=self.getParamByName(config_const.CONF_OPTIMIZER_WEIGHT_DECAY, 0)
                              )
        return None  
    
    @abstractmethod
    def getScheduler(self):
        scheduler_name = self.getParamByName(config_const.CONF_SCHEDULER_NAME)
        if scheduler_name == "WarmupLR":
            warmup_duration = self.getParamByName(config_const.CONF_OPTIMIZER_LEARNING_WARMUP_STEPS)
            warmup_init_lr = self.getParamByName(config_const.CONF_OPTIMIZER_LEARNING_WARMUP_INIT_LR)
            patience=self.getParamByName(config_const.CONF_OPTIMIZER_LEARNING_RATE_PATIENCE)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.96, patience=patience, min_lr=0.0000001)
            scheduler = WarmupLR(scheduler, init_lr=warmup_init_lr, num_warmup=warmup_duration, warmup_strategy='cos')
            return scheduler        
        elif scheduler_name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.96, patience=patience, min_lr=0.0000001)
            return scheduler  
        return None
        
   
    def get_model_params(self):
        return  {k: v for k, v in self.params.items() if k.startswith("model.")}
           
    def get_feature_size(self):
        if self.data_holder.train_data_loader:
            return self.data_holder.train_data_set[0].x_s.shape[1]
        else:
            return self.data_holder.test_data_set[0].x_s.shape[1]  

    def get_graph_builder(self):
        if self.data_holder.train_data_loader:
            return self.data_holder.train_data_set.graph_builder
        else:
            return self.data_holder.test_data_set.graph_builder

    def get_edge_feature_size(self):
        if self.data_holder.train_data_loader:
            return self.data_holder.train_data_set[0].edge_attr_s.shape[1]
        else:
            return self.data_holder.test_data_set[0].edge_attr_s.shape[1]
      
    @abstractmethod  
    def getLossFunction(self):
        loss = self.getParamByName("loss_function.name", "MSE")
        if loss == "MSE":
            return torch.nn.MSELoss()
        elif loss == "BCE":
            return torch.nn.BCELoss()
        elif loss == "CrossEntrpy":
            return torch.nn.CrossEntropyLoss()
        elif loss == "Cosine":
            self.loss_contrastive = True
            return torch.nn.CosineEmbeddingLoss(margin=0.0)
        elif loss == "Contrastive":
            self.loss_contrastive = True
            #return ContrastiveLoss()
            return None
        elif loss == "BCESiamese":
            self.loss_contrastive = True
            #return SiameseBCELoss()
            return None
        else: 
            return None
    
    
    def getParamByName(self, name, default=None, return_as_array=False):
        if name in self.params:
            if return_as_array:
                return [self.params[name]]
            else:
                return self.params[name]
        else:
            return default
        
    def setModelMetric(self, key, value):
        self.model_metrics[key] = value
    
    def count_parameters(self):
        return sum(np.prod(p.size()) for p in self.model.parameters() if p.requires_grad)

    def get_experiment_path(self):
        root = "./models/"
        if config_const.CONF_MODEL_PATH_TO_ROOT in self.params and self.params[config_const.CONF_MODEL_PATH_TO_ROOT]:
            root = self.params[config_const.CONF_MODEL_PATH_TO_ROOT]
        return os.path.join(root,self.__class__.__name__, datetime.now().strftime("%m%d_%H%M%S_%f"))

    def save_experiment_params(self):
        os.makedirs(self.experiment_path, exist_ok=True) 
        path_to_src = os.path.join(self.experiment_path, "src")
        os.makedirs(path_to_src, exist_ok=True)  
        utils.save_json(self.experiment_path, "experiment_params.json",self.params)
        self.save_classes(self, path_to_src)
        self.save_classes(self.model, path_to_src)

    def getClassPath(self, class_instance):
        return os.path.abspath(sys.modules[class_instance.__module__].__file__)

    def save_classes(self, class_name, path_to_src):
        path = self.getClassPath(class_name)
        shutil.copy(path, path_to_src)

    def train_model(self, epoch):
        self.model.train()
        loss,_,_,_, _ = self.do_training(epoch=epoch,  data_loader=self.data_holder.train_data_loader, is_training=True)
        self.print_metric(epoch, loss, self.best_train_loss,"Train")
        self.mlflow_metric(epoch, loss, "Train")
        self.update_best_metric("Train")
        return loss
        
        
    def evaluate_model(self, epoch):
        self.model.eval()
        loss, emb_a, emb_b,ls,y = self.do_training(epoch=epoch, data_loader=self.data_holder.test_data_loader, is_training=False)
        self.print_metric(epoch, loss, self.best_test_loss, "Test")
        self.mlflow_metric(epoch, loss, "Test")
        self.update_best_metric("Test")
        return loss

    def mlflow_metric(self,epoch, loss, type):
        if self.mlflow:
            self.mlflow.log_metric(key=type+"_loss", value=float(loss), step=epoch)
            for key in self.model_metrics:
                self.mlflow.log_metric(key=type+"_"+key, value=float(self.model_metrics[key]), step=epoch)

    def print_metric(self,epoch, loss, best_lost, type):
        if epoch % self.show_info_by == 0:
            logging.info(f'Epoch: {epoch:03d}/{self.epoch_number:03d} [{type}] Loss:{loss:.3f} [{round(((loss-best_lost)/best_lost)*100,1)}%] {" ".join([str(key) +":"+str(round(self.model_metrics[key],3)) for key in self.model_metrics])}')
        

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
    
    def update_best_metric(self, type):
        for key in self.model_metrics:
            new_key = type + " " + key
            if new_key not in self.best_model_metric:
                self.best_model_metric[new_key] = 0
            if self.model_metrics[key] > self.best_model_metric[new_key]:
                self.best_model_metric[new_key] = self.model_metrics[key]
        
    def fit(self, tune=None, trials_stats=None):  
        self.epoch_number=self.getParamByName("train.epoch_number", 101)
        early_stop_by = self.getParamByName("train.early_stop_by", 10)
        early_stopping_counter = 0
        for epoch in range(self.epoch_number):
            if early_stopping_counter <= early_stop_by:
                train_loss =  self.train_model(epoch)
                if train_loss < self.best_train_loss: self.best_train_loss=train_loss
                if epoch % 5 == 0 and epoch>0:
                    test_loss  = self.evaluate_model(epoch)
                    if tune and self.getParamByName("ray.enable", False):
                        tune.report(min_loss=test_loss)
                    if test_loss < self.best_test_loss: 
                        self.best_test_loss=test_loss
                        torch.save(self.model.state_dict(), os.path.join(self.experiment_path, "model.pt"))
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
            else:
                logging.info(f'Early stopping at epoch:{epoch}')
                break            
        logging.info(f'Best Train Loss: {self.best_train_loss:.4f}')
        logging.info(f'Best Test Loss: {self.best_test_loss:.4f}')
        for key in self.best_model_metric:
            logging.info(f'Best {key}: {str(round(self.best_model_metric[key],4))}')
        
        self.experiment_path
        out_statistic = {}
        out_statistic['Epoch']=str(epoch)
        out_statistic['Train loss']=str(round(self.best_train_loss,4))
        out_statistic['Test loss']=str(round(self.best_test_loss,4))
        out_statistic['trail_model_path'] = self.experiment_path
        for key in self.best_model_metric:
            out_statistic[key]=str(round(self.best_model_metric[key],4))
        with open(os.path.join(self.experiment_path, "results.json"), "w") as wf:
            json.dump(out_statistic, wf, indent=4)    
        
        return self.best_test_loss, out_statistic