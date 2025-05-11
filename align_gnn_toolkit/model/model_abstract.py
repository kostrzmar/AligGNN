import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr 
from scipy.stats import spearmanr 
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from enum import Enum
from utils import config_const
from tqdm import tqdm
import copy
from model.pooling import ASAEPooling
from model.model_GMN import GraphMatchingConvolution
from sentence_transformers.util import pairwise_cos_sim
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn import GCNConv, GINConv, GINEConv, TransformerConv, GeneralConv,  JumpingKnowledge,global_mean_pool, EdgePooling, TopKPooling
from torch_geometric.nn import GATv2Conv, ResGatedGraphConv, AGNNConv, TAGConv, SGConv, RGATConv, SAGEConv, ASAPooling, SAGPooling, GraphConv, GATConv
from torch_geometric.nn import ResGatedGraphConv, CGConv, GENConv
from torch_geometric.nn import MultiAggregation, SoftmaxAggregation

DISPLAY_INFO_EVERY = 0.005 # every 1 / 100 batch
INCLUDE_SOFT_F1 = False


class SiameseDistanceMetric(Enum):

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)
    HAMMING_SIM = lambda x, y : torch.mean(torch.tanh(x) * torch.tanh(y), dim=1)

def dot_score(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


def pairwise_angle_sim(x: torch.Tensor, y: torch.Tensor):
    # https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py
    #x = _convert_to_tensor(x)
    #y = _convert_to_tensor(y)

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


def calculate_loss(pred_logits, gold, emb_s=None, emb_t=None, loss_type=None):
    loss = 0.0
    if loss_type == "BCE":
        pos_weight = torch.FloatTensor([50]) # was 4 or 50
        loss_raw = F.binary_cross_entropy_with_logits(pred_logits, gold,reduction="none")
        weight = torch.ones_like(loss_raw)
        weight[gold==1.] = pos_weight
        loss = ((loss_raw) * weight).mean() #* 20 #+ 1 - soft_f1 # was 20   
    elif loss_type == "COS":
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
        labels = labels[:, None] < labels[None, :]
        labels = labels.float()
        # mask out irrelevant pairs so they are negligible after exp()
        scores = scores - (1 - labels) * 1e12
        # append a zero as e^0 = 1
        scores = torch.cat((torch.zeros(1).to(scores.device), scores.view(-1)), dim=0)
        loss = torch.logsumexp(scores, dim=0)
    elif loss_type =="AnglELoss":
        #from -> https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/CoSENTLoss.py#L13-L114
        scale: float = 20.0
        scores = pairwise_angle_sim(emb_s, emb_t)
        scores = scores * scale
        # label matrix indicating which pairs are relevant
        labels = labels[:, None] < labels[None, :]
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
    elif loss_type =="MSE":
        y_pred = torch.sigmoid(pred_logits)
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
    elif loss_type =="hamming":
        approximate_hamming_similarity  = torch.mean(torch.tanh(emb_s) * torch.tanh(emb_t), axis=1)
        loss = 0.25 * (gold.float() - approximate_hamming_similarity)**2
    return loss
 
def do_loss(pred_logits, gold, emb_s=None, emb_t=None, loss_types=['COS'], loss_coefficients=[1.0]):
    soft_f1 = 0
    if INCLUDE_SOFT_F1:
        if "BCE"  in loss_types:
            pass
        else:
            y_pred = F.relu(F.cosine_similarity(emb_s, emb_t, dim=-1))
        tp = torch.sum(torch.mul(gold, y_pred))
        fp = torch.sum(torch.mul(1 - gold, y_pred))
        fn = torch.sum(torch.mul(gold, 1 - y_pred))
        soft_f1 = (2 * tp) / (2 * tp + fn + fp + 1e-16)
        soft_f1 = 1 - soft_f1
    final_loss = soft_f1
    for loss_type, loss_coefficient in zip(loss_types, loss_coefficients): 
        loss = calculate_loss(pred_logits, gold, emb_s, emb_t, loss_type)
        final_loss+= loss*loss_coefficient

    return final_loss




def train(model, train_loader, optimizer, device):
    model.train()
    loss_all = 0
    storage = get_storage(['mse', 'p', 's', 'loss', 'loss_all'])
    dataset_size  = len(train_loader)
    show_every = int(dataset_size * DISPLAY_INFO_EVERY)
    if show_every ==0: show_every =1
    if config_const.CONF_TRAIN_LOSS_FUNCTION_NAME in model.params:
        LOSS_TYPE  = model.params[config_const.CONF_TRAIN_LOSS_FUNCTION_NAME]
    for index, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        y_pred_logits, emb_s,emb_t, l, e = model(data)
        loss = do_loss(y_pred_logits, data.y, emb_s, emb_t, LOSS_TYPE)
        loss = loss + l + e
        if index % show_every ==0:
            y_pred = torch.sigmoid(y_pred_logits)
            display_progress_info(index,dataset_size, loss, loss_all, y_pred.detach().cpu(), data.y.detach().cpu(), storage, 'T')
        loss.backward()
        optimizer.step()
        loss_all += loss.item() * data.num_graphs
    return loss_all / len(train_loader.dataset)

def eval(model, loader, device, type):
        model.eval()
        error = 0
        loss_all = 0
        storage = get_storage(['mse', 'p', 's', 'loss', 'loss_all'])
        total_size = len(loader.dataset)
        local_model_metrics = {}
        g_y_all = []
        p_y_all = []
        dataset_size = len(loader)
        show_every = int(dataset_size * DISPLAY_INFO_EVERY)
        if config_const.CONF_TRAIN_LOSS_FUNCTION_NAME in model.params:
            LOSS_TYPE  = model.params[config_const.CONF_TRAIN_LOSS_FUNCTION_NAME]
        if show_every ==0: show_every =1
        for index, data in enumerate(loader):
            data = data.to(device)
            with torch.no_grad():
                y_pred_logits, emb_s,emb_t, l, e = model(data)
                g_y=  (data.y).float().detach().cpu().numpy()
                y_pred = torch.sigmoid(y_pred_logits)
                p_y= y_pred.float().detach().cpu().numpy()
                g_y_all.extend(g_y)
                p_y_all.extend(p_y)
                loss = do_loss(y_pred_logits, data.y, emb_s,emb_t, LOSS_TYPE)
                loss = loss + l + e
                if index % show_every ==0:
                    display_progress_info(index,dataset_size, loss, loss_all, y_pred.detach().cpu(), data.y.detach().cpu(), storage, 'V')
                loss_all += loss.item() * data.num_graphs       
        local_model_metrics = {
                        "type":type,
                        "loss": loss_all / total_size,
                        "R2_score":r2_score(g_y_all , p_y_all),
                        "MQE":mean_squared_error(g_y_all , p_y_all),
                        "MAE":mean_absolute_error(g_y_all , p_y_all),
                        "RMQE" : np.sqrt(mean_squared_error(g_y_all , p_y_all)),
                        "pearsonr":pearsonr(g_y_all , p_y_all)[0],
                        "SpearmanR":spearmanr(g_y_all , p_y_all)[0]
                        }
                
        return loss_all / total_size,local_model_metrics

def predict(model, test_data_loader, device, embedding_as_numpy=True):
    model.eval()
    predict_y = []
    gold_y = []
    emb1 = []
    emb2 = []
    model.to(device)
    for index, batch in enumerate(tqdm(test_data_loader)):
        graph = batch
        graph = graph.to(device)
        gold_y.extend((graph.y).detach().cpu())
        with torch.no_grad():
            y_pred_logits,emb_src,emb_trg, l, e =model(graph)
            y_pred = torch.sigmoid(y_pred_logits)
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



def get_storage(items):
    storage = {}
    for item in items:
        storage[item] = {}
        storage[item]['min'] = None
        storage[item]['max'] = None
        storage[item]['current'] = None
    return storage

def add_to_storage(storage, type, item):
    storage[type]['current'] = item
    min = storage[type]['min']
    max = storage[type]['max']
    if not max or item > max:
        storage[type]['max'] = item
    if not min or item < min:
        storage[type]['min'] = item
        
def get_from_storage(storage, type):
    return f'{storage[type]["current"]:.04f}|{storage[type]["min"]:.03f}|{storage[type]["max"]:.03f}' 

def display_progress_info(index,dataset_size, loss, loss_all, y_pred, y_gold, storage=None, phase=None):
        if storage:
            add_to_storage(storage, "mse", mean_squared_error(y_pred, y_gold))
            add_to_storage(storage,"p", pearsonr(y_pred, y_gold)[0])
            add_to_storage(storage, "s", spearmanr(y_pred, y_gold)[0])
            add_to_storage(storage, "loss", loss)
            print(f'{phase}_I:{(index/dataset_size)*100:.02f}% L:{get_from_storage(storage, "loss")} LA:{loss_all:.04f} M:{get_from_storage(storage, "mse")} P:{get_from_storage(storage, "p")} S:{get_from_storage(storage, "s")} ', end='\r')   
        else:    
            print(f'Train Index: {(index/dataset_size)*100:.02f}% Loss: {loss:.04f} LossAll: {loss_all:.04f}', end='\r')
            
            

class AbstractModel(torch.nn.Module):

    def __init__(self, params, number_of_labels, edge_feature_size=None, vocab=None, embed_one_hot=None):
        super(AbstractModel, self).__init__()
        self.params = params
        self.number_labels = number_of_labels
        self.edge_feature_size = edge_feature_size
        self.vocab = vocab        

        
        self.dropout = self.get_config_value(config_const.CONF_MODEL_DROPOUT_RATE, 0)    
        self.gnn_operator = self.get_config_value(config_const.CONF_MODEL_SIMGNN_GNN_OPERATOR, "GCNConv")
        self.pooling_gnn = self.get_config_value(config_const.CONF_MODEL_POOL_GNN_OPERATOR, "GCNConv")
        self.number_of_layers  = self.get_config_value(config_const.CONF_MODEL_CONV_LAYERS_NBR, 3)
        self.do_skip_connection  = self.get_config_value(config_const.CONF_MODEL_CONV_DO_SKIP_CONNECTION, True)
        self.activation_type = self.get_config_value(config_const.CONF_MODEL_CONV_ACTIVATION_TYPE, "relu")
        self.scoring_type = self.get_config_value(config_const.CONF_MODEL_SCORING_TYPE, "simple_activation_batch")
        self.read_out_type = self.get_config_value(config_const.CONF_MODEL_READ_OUT_TYPE, "mean")
        
        self.filters_1 = self.get_config_value(config_const.CONF_MODEL_SIMGNN_FILTERS_1, 1200)
        self.filters_2 = self.get_config_value(config_const.CONF_MODEL_SIMGNN_FILTERS_2, 900 )
        self.filters_3 = self.get_config_value(config_const.CONF_MODEL_SIMGNN_FILTERS_3, 300 )
        
        self.has_two_convolutions =False
        if isinstance(self.gnn_operator, list):
            self.has_two_convolutions = True
        self.include_abs = True
        self.final_nbr_of_concatenation =2 
        if self.include_abs:
            self.final_nbr_of_concatenation = 3
        self.ratio=0.8
        self.apply_pooling = False
        self.include_edge = False
        self.feature_count = None
        self.static_nbr_of_feature = None
        self.skip_connection_feature_multiplyer = 0
        self.skip_connection_feature_multiplyer_b = 0
        self.number_heads = 1
        self.apply_heads=False 
        self.cross_conv  = False
        self.cross_conv_2 = False
        self.embed_one_hot_nodes = None
        self.embed_one_hot_edges = None
        self.include_softmax_aggr = self.get_config_value(config_const.CONF_MODEL_SOFT_MAX_AGGR, False)
        self.softmax_aggr_multiplyer = 1
        if self.include_softmax_aggr:
            self.softmax_aggr_multiplyer = 3
        self.embed_one_hot = self.get_config_value(config_const.CONF_MODEL_EMBED_ONE_HOT, False)
        self.one_hot_embed_size = self.get_config_value(config_const.CONF_MODEL_ONE_HOT_EMBEDDING_SIZE)    
        if self.embed_one_hot and embed_one_hot is not None:
            self.embed_one_hot_nodes = embed_one_hot[0]
            self.embed_one_hot_edges = embed_one_hot[1]
            
            self.embed_one_hot_nodes_active = [x for x in self.embed_one_hot_nodes if x[0]] 
            self.embed_one_hot_edges_active = [x for x in self.embed_one_hot_edges if x[0]]       
        
        self.setup_layers()
        self.setup_scoring()
        self.reset_parameters()
        

    def get_config_value(self, name, default_value=None):
        if name in self.params:
            return self.params[name]
        if default_value:
            return default_value
        return None

    def setup_scoring(self):

        self.feature_count = self.filters_3 * self.final_nbr_of_concatenation
        if self.apply_heads:
           self.feature_count = self.feature_count * self.number_heads
           pass

        if self.scoring_type =="3layers":

            if self.static_nbr_of_feature:
                self.fully_connected_first = torch.nn.Linear(self.final_nbr_of_concatenation * self.static_nbr_of_feature* (self.skip_connection_feature_multiplyer +self.skip_connection_feature_multiplyer_b), int(self.feature_count/2))
            else:
                self.fully_connected_first = torch.nn.Linear(self.feature_count * (self.skip_connection_feature_multiplyer +self.skip_connection_feature_multiplyer_b), int(self.feature_count/2))
            self.fully_connected_second = torch.nn.Linear(int(self.feature_count/2), int(self.feature_count/4))
            self.fully_connected_third = torch.nn.Linear(int(self.feature_count/4), int(self.feature_count/6))
            self.scoring_layer = torch.nn.Linear(int(self.feature_count/6), 1)        
        elif self.scoring_type =="simple_activation_batch":
            
            if self.static_nbr_of_feature:
                self.scoring = torch.nn.Sequential(
                            torch.nn.Linear(self.final_nbr_of_concatenation * self.static_nbr_of_feature * (self.skip_connection_feature_multiplyer +self.skip_connection_feature_multiplyer_b), self.filters_3),
                            torch.nn.BatchNorm1d(self.filters_3),
                            torch.nn.ReLU(),
                            torch.nn.Linear(self.filters_3, 1)  
                )         
            else:
                self.scoring = torch.nn.Sequential(
                            torch.nn.Linear(self.feature_count * (self.skip_connection_feature_multiplyer +self.skip_connection_feature_multiplyer_b), self.filters_3),
                            torch.nn.BatchNorm1d(self.filters_3),
                            torch.nn.ReLU(),
                            torch.nn.Linear(self.filters_3, 1)
                )

    def setup_layers(self): 
        filters_1 = self.filters_1
        filters_2 = self.filters_2
        filters_3 = self.filters_3
        
        if self.embed_one_hot:
            self.one_hot_embed_nodes_embeddings = torch.nn.ModuleList()
            self.one_hot_embed_edges_embeddings = torch.nn.ModuleList()
            for node in self.embed_one_hot_nodes_active:
                self.one_hot_embed_nodes_embeddings.append(torch.nn.Linear(node[2], self.one_hot_embed_size, bias=False, device="cuda:0"))
            for edge in self.embed_one_hot_edges_active:
                self.one_hot_embed_edges_embeddings.append(torch.nn.Linear(edge[2], self.one_hot_embed_size, bias=False, device="cuda:0"))
            
        if self.has_two_convolutions:
            self.convolution_1, self.convolution_2, self.convolution_3, self.skip_connection_feature_multiplyer = self.initialize_convs(self.gnn_operator[0], filters_1, filters_2, filters_3)
            self.convolution_1b, self.convolution_2b, self.convolution_3b, self.skip_connection_feature_multiplyer_b = self.initialize_convs(self.gnn_operator[1], filters_1, filters_2, filters_3)
            if self.static_nbr_of_feature:
                filters_1 = self.static_nbr_of_feature
                filters_2 = self.static_nbr_of_feature
                filters_3 = self.static_nbr_of_feature

            if self.do_skip_connection:
                self.skip_conn_1 = torch.nn.Linear(self.number_labels, filters_1 * self.skip_connection_feature_multiplyer * self.softmax_aggr_multiplyer)
                self.skip_conn_2 = torch.nn.Linear(filters_1* self.skip_connection_feature_multiplyer, filters_2 * self.skip_connection_feature_multiplyer * self.softmax_aggr_multiplyer)
                self.skip_conn_3 = torch.nn.Linear(filters_2* self.skip_connection_feature_multiplyer, filters_3 * self.skip_connection_feature_multiplyer * self.softmax_aggr_multiplyer)
                self.skip_conn_1b = torch.nn.Linear(self.number_labels, filters_1 * self.skip_connection_feature_multiplyer_b * self.softmax_aggr_multiplyer)
                self.skip_conn_2b = torch.nn.Linear(filters_1* self.skip_connection_feature_multiplyer_b, filters_2 * self.skip_connection_feature_multiplyer_b * self.softmax_aggr_multiplyer)
                self.skip_conn_3b = torch.nn.Linear(filters_2* self.skip_connection_feature_multiplyer_b, filters_3 * self.skip_connection_feature_multiplyer_b * self.softmax_aggr_multiplyer)
            else:
                self.skip_conn_1 = None
                self.skip_conn_2 = None
                self.skip_conn_3 = None
                self.skip_conn_1b = None
                self.skip_conn_2b = None
                self.skip_conn_3b = None        
        else:
            self.convolution_1, self.convolution_2, self.convolution_3, self.skip_connection_feature_multiplyer = self.initialize_convs(self.gnn_operator, filters_1, filters_2, filters_3)
            
            if self.static_nbr_of_feature:
                filters_1 = self.static_nbr_of_feature
                filters_2 = self.static_nbr_of_feature
                filters_3 = self.static_nbr_of_feature
            if self.do_skip_connection:
                self.skip_conn_1 = torch.nn.Linear(self.number_labels, filters_1* self.skip_connection_feature_multiplyer*self.softmax_aggr_multiplyer)
                self.skip_conn_2 = torch.nn.Linear(filters_1* self.skip_connection_feature_multiplyer, filters_2* self.skip_connection_feature_multiplyer*self.softmax_aggr_multiplyer)
                self.skip_conn_3 = torch.nn.Linear(filters_2* self.skip_connection_feature_multiplyer, filters_3* self.skip_connection_feature_multiplyer*self.softmax_aggr_multiplyer)
            else:
                self.skip_conn_1 = None
                self.skip_conn_2 = None
                self.skip_conn_3 = None

        pooling_gnn = GCNConv
        if self.pooling_gnn =="GraphConv":
            pooling_gnn = GraphConv
        elif self.pooling_gnn =="GATConv":
            pooling_gnn = GATConv
        elif self.pooling_gnn =="GATConv":
            pooling_gnn = SAGEConv                
        if self.read_out_type =="asa_pooling": 
            
            if  self.include_edge:
                self.pooling_1 = ASAEPooling(self.filters_1*self.number_heads, self.ratio, dropout=self.dropout,  edge_dim=self.edge_feature_size, GNN=pooling_gnn)
                self.pooling_2 = ASAEPooling(self.filters_2*self.number_heads, self.ratio, dropout=self.dropout,  edge_dim=self.edge_feature_size, GNN=pooling_gnn)
                self.pooling_3 = ASAEPooling(self.filters_3*self.number_heads, self.ratio, dropout=self.dropout,  edge_dim=self.edge_feature_size, GNN=pooling_gnn)
            else:
                self.pooling_1 = ASAPooling(self.filters_1*self.number_heads, self.ratio, dropout=self.dropout, GNN=pooling_gnn)
                self.pooling_2 = ASAPooling(self.filters_2*self.number_heads, self.ratio, dropout=self.dropout, GNN=pooling_gnn)
                self.pooling_3 = ASAPooling(self.filters_3*self.number_heads, self.ratio, dropout=self.dropout, GNN=pooling_gnn)                
            self.jump = JumpingKnowledge(mode='cat')
            self.pool_lin1 = torch.nn.Linear(self.filters_1*self.number_heads + self.filters_2*self.number_heads + self.filters_3*self.number_heads, self.filters_3)
            self.apply_pooling = True
            
        elif self.read_out_type=="sag_pooling":
            self.pooling_1 = SAGPooling(self.filters_1*self.number_heads, self.ratio, GNN=pooling_gnn)
            self.pooling_2 = SAGPooling(self.filters_2*self.number_heads, self.ratio, GNN=pooling_gnn)
            self.pooling_3 = SAGPooling(self.filters_3*self.number_heads, self.ratio, GNN=pooling_gnn)                
            self.jump = JumpingKnowledge(mode='cat')
            self.pool_lin1 = torch.nn.Linear(self.filters_1*self.number_heads + self.filters_2*self.number_heads + self.filters_3*self.number_heads, self.filters_3)
            self.apply_pooling = True
        
        elif self.read_out_type=="edge_pooling":
            self.pooling_1 = EdgePooling(self.filters_1*self.number_heads )
            self.pooling_2 = EdgePooling(self.filters_2*self.number_heads )
            self.pooling_3 = EdgePooling(self.filters_3*self.number_heads)                
            self.jump = JumpingKnowledge(mode='cat')
            self.pool_lin1 = torch.nn.Linear(self.filters_1*self.number_heads + self.filters_2*self.number_heads + self.filters_3*self.number_heads, self.filters_3)
            self.apply_pooling = True
        
        elif self.read_out_type=="topk_pooling":
            self.pooling_1 = TopKPooling(self.filters_1*self.number_heads, self.ratio)
            self.pooling_2 = TopKPooling(self.filters_2*self.number_heads, self.ratio)
            self.pooling_3 = TopKPooling(self.filters_3*self.number_heads, self.ratio)                
            self.jump = JumpingKnowledge(mode='cat')
            self.pool_lin1 = torch.nn.Linear(self.filters_1*self.number_heads + self.filters_2*self.number_heads + self.filters_3*self.number_heads, self.filters_3)
            self.apply_pooling = True
        
        
            

    def reset_parameters_in_seq(self, sequence):
        for item in sequence:
            if hasattr(item, 'reset_parameters') and callable(item.reset_parameters):
                item.reset_parameters()

    def reset_parameters(self):
        self.convolution_1.reset_parameters()
        self.convolution_2.reset_parameters()
        self.convolution_3.reset_parameters()
        if self.do_skip_connection:
            self.skip_conn_1.reset_parameters()
            self.skip_conn_2.reset_parameters()
            self.skip_conn_3.reset_parameters()
        
        if self.has_two_convolutions:
            self.convolution_1b.reset_parameters()
            self.convolution_2b.reset_parameters()
            self.convolution_3b.reset_parameters()
            if self.do_skip_connection:
                self.skip_conn_1b.reset_parameters()
                self.skip_conn_2b.reset_parameters()
                self.skip_conn_3b.reset_parameters()
        
        if self.embed_one_hot:
            for emb_nodes in self.one_hot_embed_nodes_embeddings:
                emb_nodes.reset_parameters()
            for emb_edge in self.one_hot_embed_edges_embeddings:
                emb_edge.reset_parameters()
                
        if self.scoring_type == '3layers':
            self.fully_connected_first.reset_parameters()
            self.fully_connected_second.reset_parameters()
            self.fully_connected_third.reset_parameters()
            self.scoring_layer.reset_parameters()

        elif self.scoring_type == 'simple_activation_batch':
            self.reset_parameters_in_seq(self.scoring)
        
        if self.apply_pooling:    
            self.pooling_1.reset_parameters()
            self.pooling_2.reset_parameters()
            self.pooling_3.reset_parameters()
            self.pool_lin1.reset_parameters()
                
    def lock_convolution(self):
        def _lock(convolution):
            for param in convolution.parameters():
                param.requires_grad_(False)
        _lock(self.convolution_1)
        _lock(self.convolution_2)
        _lock(self.convolution_3)
        if self.has_two_convolutions:
            _lock(self.convolution_1b)
            _lock(self.convolution_2b)
            _lock(self.convolution_3b)    
    
    def getMLP(self, in_channels, hidden_channels, version=0):
        if version==0:
            return torch.nn.Sequential(
                torch.nn.Linear(in_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels, track_running_stats = False),
            )
        elif version==1:
            return torch.nn.Sequential(
                    torch.nn.Linear(in_channels, hidden_channels),
                    torch.nn.BatchNorm1d(hidden_channels, track_running_stats = False),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.BatchNorm1d(hidden_channels, track_running_stats = False),
                    torch.nn.LeakyReLU(),
                )
            
    def initialize_convs(self, gnn_operator, filters_1, filters_2,filters_3):
        skip_connection_feature_multiplyer =  1
        self.static_nbr_of_feature = None
        convolution_1, convolution_2, convolution_3 = None, None, None
        if gnn_operator == "GCNConv":
            self.include_edge = False
            convolution_1 = GCNConv(self.number_labels, filters_1)
            convolution_2 = GCNConv(filters_1, filters_2)
            convolution_3 = GCNConv(filters_2, filters_3)
            
        elif gnn_operator == "GeneralConv":
            self.include_edge = True
            convolution_1 = GeneralConv(self.number_labels, filters_1, in_edge_channels=self.edge_feature_size, l2_normalize=False)
            convolution_2 = GeneralConv(filters_1, filters_2, in_edge_channels=self.edge_feature_size, l2_normalize=False)
            convolution_3 = GeneralConv(filters_2, filters_3, in_edge_channels=self.edge_feature_size, l2_normalize=False)    
        elif gnn_operator ==  "GATv2Conv":
            self.include_edge = True
            number_heads = 4
            dropout = 0.6
            skip_connection_feature_multiplyer = number_heads
            convolution_1 = GATv2Conv(self.number_labels, filters_1, edge_dim=self.edge_feature_size, heads=number_heads, dropout=dropout)
            convolution_2 = GATv2Conv(filters_1*number_heads, filters_2, edge_dim=self.edge_feature_size, heads=number_heads, dropout=dropout)
            convolution_3 = GATv2Conv(filters_2*number_heads, filters_3, edge_dim=self.edge_feature_size, heads=number_heads, dropout=dropout)
        elif gnn_operator == "GINConv":
            self.include_edge = False
            convolution_1 = GINConv(self.getMLP(self.number_labels, filters_1, version=1), train_eps=True)
            convolution_2 = GINConv(self.getMLP(filters_1, filters_2, version=1), train_eps=True)
            convolution_3 = GINConv(self.getMLP(filters_2, filters_3, version=1), train_eps=True)
        elif gnn_operator == "GINEConv":
            self.include_edge = True
            convolution_1 = GINEConv(self.getMLP(self.number_labels, filters_1, version=1), edge_dim=self.edge_feature_size, train_eps=True)
            convolution_2 = GINEConv(self.getMLP(filters_1, filters_2, version=1), edge_dim=self.edge_feature_size, train_eps=True)
            convolution_3 = GINEConv(self.getMLP(filters_2, filters_3, version=1), edge_dim=self.edge_feature_size, train_eps=True)
        elif gnn_operator == "ResGatedGraphConv":
            self.include_edge = True
            convolution_1 = ResGatedGraphConv(self.number_labels, filters_1, edge_dim=self.edge_feature_size )
            convolution_2 = ResGatedGraphConv(filters_1, filters_2, edge_dim=self.edge_feature_size )
            convolution_3 = ResGatedGraphConv(filters_2, filters_3, edge_dim=self.edge_feature_size)         
        elif gnn_operator== "GENConv":
            self.include_edge = True
            convolution_1 = GENConv(self.number_labels, filters_1, edge_dim=self.edge_feature_size, learn_t=True )
            convolution_2 = GENConv(filters_1, filters_2, edge_dim=self.edge_feature_size, learn_t=True )
            convolution_3 = GENConv(filters_2, filters_3, edge_dim=self.edge_feature_size, learn_t=True)               
        elif gnn_operator == "GATEConv":
            self.include_edge = False
            convolution_1 = ResGatedGraphConv(self.number_labels, filters_1, edge_dim=self.edge_feature_size)
            convolution_2 = ResGatedGraphConv(filters_1, filters_2, edge_dim=self.edge_feature_size)
            convolution_3 = ResGatedGraphConv(filters_2, filters_3, edge_dim=self.edge_feature_size)    
        elif gnn_operator == "AGNNConv":
            self.include_edge = False
            self.static_nbr_of_feature  = self.number_labels
            convolution_1 = AGNNConv(requires_grad=False)
            convolution_2 = AGNNConv(requires_grad=False)
            convolution_3 = AGNNConv(requires_grad=False)   
        elif gnn_operator == "TAGConv":
            self.include_edge = False
            convolution_1 = TAGConv(self.number_labels, filters_1)
            convolution_2 = TAGConv(filters_1, filters_2)
            convolution_3 = TAGConv(filters_2, filters_3)               
        elif gnn_operator == "SGConv":
            self.include_edge = False
            convolution_1 = SGConv(self.number_labels, filters_1, K =3)
            convolution_2 = SGConv(filters_1, filters_2, K=3)
            convolution_3 = SGConv(filters_2, filters_3, K=3)        
        elif gnn_operator == "RGATConv":
            self.include_edge = True
            convolution_1 = RGATConv(self.number_labels, filters_1, self.edge_feature_size)
            convolution_2 = RGATConv(filters_1, filters_2, self.edge_feature_size)
            convolution_3 = RGATConv(filters_2, filters_3, self.edge_feature_size)
        elif gnn_operator == "SAGEConv":
            self.include_edge = False
            if self.include_softmax_aggr:
                aggr = MultiAggregation (
                    [SoftmaxAggregation(t=0.01, learn=True),
                    SoftmaxAggregation(t=1, learn=True),
                    SoftmaxAggregation(t=100, learn=True)])            
                convolution_1 = SAGEConv(self.number_labels, filters_1, aggr=aggr, aggr_kwargs=None)
                convolution_2 = SAGEConv(filters_1, filters_2, aggr=copy.deepcopy(aggr), aggr_kwargs=None)
                convolution_3 = SAGEConv(filters_2, filters_3, aggr=copy.deepcopy(aggr), aggr_kwargs=None)
            else:
                convolution_1 = SAGEConv(self.number_labels, filters_1)
                convolution_2 = SAGEConv(filters_1, filters_2)
                convolution_3 = SAGEConv(filters_2, filters_3)    
        elif gnn_operator == "CGConv":
            self.include_edge = True
            convolution_1 = CGConv(self.number_labels,  self.edge_feature_size)
            convolution_2 = CGConv(self.number_labels,  self.edge_feature_size)
            convolution_3 = CGConv(self.number_labels,  self.edge_feature_size)                   
        elif gnn_operator == "GraphMatchingConv":
            self.include_edge = False
            self.cross_conv  = True
            convolution_1 = GraphMatchingConvolution(self.number_labels, filters_1)
            convolution_2 = GraphMatchingConvolution(filters_1, filters_2)
            convolution_3 = GraphMatchingConvolution(filters_2, filters_3)

        elif gnn_operator == "TransformerConv":
            self.include_edge = True
            self.number_heads = 4
            self.apply_heads=True
            beta = True
            dropout = 0.0
            concat = True
            convolution_1 = TransformerConv(self.number_labels, filters_1, edge_dim=self.edge_feature_size, heads=self.number_heads, dropout=dropout, concat=concat, beta=beta)
            convolution_2 = TransformerConv(filters_1*self.number_heads, filters_2, edge_dim=self.edge_feature_size, heads=self.number_heads, dropout=dropout, concat=concat, beta=beta)
            convolution_3 = TransformerConv(filters_2*self.number_heads, filters_3, edge_dim=self.edge_feature_size, heads=self.number_heads, dropout=dropout, concat=concat, beta=beta)   
        else:
            raise NotImplementedError("Unknown GNN-Operator.")
        return convolution_1, convolution_2, convolution_3, skip_connection_feature_multiplyer
    
    def lock_convolution(self):
        for param in self.convolution_1.parameters():
            param.requires_grad_(False)
        for param in self.convolution_2.parameters():
            param.requires_grad_(False)
        for param in self.convolution_3.parameters():
            param.requires_grad_(False)
        if self.do_skip_connection:
            for param in self.skip_conn_1.parameters():
                param.requires_grad_(False)
            for param in self.skip_conn_2.parameters():
                param.requires_grad_(False)
            for param in self.skip_conn_3.parameters():
                param.requires_grad_(False)
        
    def cross_convolution_pass(self, data):
        x_s, edge_index_s, edge_attr_s, s_batch = data.x_s, data.edge_index_s, data.edge_attr_s, data.x_s_batch
        x_t, edge_index_t, edge_attr_t, t_batch = data.x_t, data.edge_index_t, data.edge_attr_t, data.x_t_batch
        
        x_s, edge_attr_s = self.do_embed_one_hot(x_s, edge_attr_s)
        x_t, edge_attr_t = self.do_embed_one_hot(x_t, edge_attr_t)
        
        x_s, edge_index_s, s_batch, x_t, edge_index_t, t_batch = self.convolution_1(x_s, edge_index_s, s_batch, x_t, edge_index_t, t_batch)           

        x_s = self.activate(x_s, self.activation_type)
        x_t = self.activate(x_t, self.activation_type)
        if self.apply_pooling: 
            xs_s = [global_mean_pool(x_s, s_batch)]
            xs_t = [global_mean_pool(x_t, t_batch)]
            
        x_s = F.dropout(x_s, p=self.dropout, training=self.training)
        x_t = F.dropout(x_t, p=self.dropout, training=self.training)
        
        if self.apply_pooling:
            x_s, edge_index_s, edge_attr_s, s_batch = self.do_pooling(self.pooling_1, x=x_s, edge_index=edge_index_s,edge_attr=edge_attr_s, batch=s_batch) 
            x_t, edge_index_t, edge_attr_t, t_batch = self.do_pooling(self.pooling_1, x=x_t, edge_index=edge_index_t,edge_attr=edge_attr_t, batch=t_batch) 
        
        if self.number_of_layers >=2:
            x_s, edge_index_s, s_batch, x_t, edge_index_t, t_batch = self.convolution_2(x_s, edge_index_s, s_batch, x_t, edge_index_t, t_batch)           

            x_s = self.activate(x_s, self.activation_type)
            x_t = self.activate(x_t, self.activation_type)
            if self.apply_pooling: 
                xs_s += [global_mean_pool(x_s, s_batch)]
                xs_t += [global_mean_pool(x_t, t_batch)]
            x_s = F.dropout(x_s, p=self.dropout, training=self.training)
            x_t = F.dropout(x_t, p=self.dropout, training=self.training)
            if self.apply_pooling:
                x_s, edge_index_s, edge_attr_s, s_batch = self.do_pooling(self.pooling_2, x=x_s, edge_index=edge_index_s,edge_attr=edge_attr_s, batch=s_batch) 
                x_t, edge_index_t, edge_attr_t, t_batch = self.do_pooling(self.pooling_2, x=x_t, edge_index=edge_index_t,edge_attr=edge_attr_t, batch=t_batch) 
        
        if self.number_of_layers >=3:
            x_s, edge_index_s, s_batch, x_t, edge_index_t, t_batch = self.convolution_3(x_s, edge_index_s, s_batch, x_t, edge_index_t, t_batch)           

            x_s = self.activate(x_s, self.activation_type)
            x_t = self.activate(x_t, self.activation_type)
            if self.apply_pooling: 
                xs_s += [global_mean_pool(x_s, s_batch)]
                xs_t += [global_mean_pool(x_t, t_batch)]
                x_s, edge_index_s, edge_attr_s, s_batch = self.do_pooling(self.pooling_3, x=x_s, edge_index=edge_index_s,edge_attr=edge_attr_s, batch=s_batch) 
                x_t, edge_index_t, edge_attr_t, t_batch = self.do_pooling(self.pooling_3, x=x_t, edge_index=edge_index_t,edge_attr=edge_attr_t, batch=t_batch) 
        if self.apply_pooling:     
            x_s = self.jump(xs_s)
            x_s = F.relu(self.pool_lin1(x_s))  
            x_t = self.jump(xs_t)
            x_t = F.relu(self.pool_lin1(x_t))             
            
        return x_s,x_t



    def do_embed_one_hot(self, features, edge_attr):
        if self.embed_one_hot:
            node_emb_index = 0
            edge_emb_index = 0
            node_emb = []
            edge_emb = []
            for nodes in self.embed_one_hot_nodes:
                node_x = features[:, nodes[1]:nodes[1]+nodes[2]]
                if nodes[0]:
                    node_x = self.one_hot_embed_nodes_embeddings[node_emb_index](node_x)
                    node_emb_index+=1
                node_emb.append(node_x)
            for edges in self.embed_one_hot_edges:
                edges_x = edge_attr[:, edges[1]:edges[1]+edges[2]]
                if edges[0]:
                    edges_x = self.one_hot_embed_edges_embeddings[edge_emb_index](edges_x)
                    edge_emb_index+=1
                edge_emb.append(edges_x)
            
            features = torch.cat(node_emb, dim=-1)
            edge_attr = torch.cat(edge_emb, dim=-1) 
        return features, edge_attr

    def convolutional_pass(self, edge_index, features, edge_attr, batch):
        features, edge_attr = self.do_embed_one_hot(features, edge_attr)
        if self.include_edge:
            if self.has_two_convolutions:             
                features_a=self.do_conv_pass_with_edge(self.convolution_1,self.convolution_2, self.convolution_3, self.skip_conn_1, self.skip_conn_2, self.skip_conn_3, edge_index, features, edge_attr, self.dropout, batch)
                features_b=self.do_conv_pass_with_edge(self.convolution_1b,self.convolution_2b, self.convolution_3b, self.skip_conn_1b, self.skip_conn_2b, self.skip_conn_3b, edge_index, features, edge_attr, self.dropout, batch)
                features=torch.cat((features_a, features_b), dim=1)
            else:
                features=self.do_conv_pass_with_edge(self.convolution_1,self.convolution_2, self.convolution_3, self.skip_conn_1, self.skip_conn_2, self.skip_conn_3, edge_index, features, edge_attr, self.dropout, batch)
        else:
            if self.has_two_convolutions:
                features_a=self.do_conv_pass_without_edge(self.convolution_1,self.convolution_2, self.convolution_3, self.skip_conn_1, self.skip_conn_2, self.skip_conn_3, edge_index, features, self.dropout, batch)
                features_b=self.do_conv_pass_without_edge(self.convolution_1b,self.convolution_2b, self.convolution_3b, self.skip_conn_1b, self.skip_conn_2b, self.skip_conn_3b, edge_index, features, self.dropout, batch)
                features=torch.cat((features_a, features_b), dim=1)
            else:
                features=self.do_conv_pass_without_edge(self.convolution_1,self.convolution_2, self.convolution_3, self.skip_conn_1, self.skip_conn_2, self.skip_conn_3, edge_index, features, self.dropout, batch)
            
        return features

    def do_conv(self, convolution, skip_conn, features, edge_index, edge_attr=None):                
        if self.do_skip_connection:
            if edge_attr is not None:
                return convolution(features, edge_index,edge_attr) + skip_conn(features)
            else:
                return convolution(features, edge_index) + skip_conn(features)
        else:
            if edge_attr is not None:
                return convolution(features, edge_index,edge_attr) 
            else:
                return convolution(features, edge_index) 
            

    def do_conv_pass_with_edge(self, convolution_1, convolution_2, convolution_3, skip_conn_1, skip_conn_2, skip_conn_3, edge_index, features, edge_attr, dropout, batch):
        features = self.do_conv(convolution_1, skip_conn_1, features, edge_index,edge_attr)
        features = self.activate(features, self.activation_type)
        if self.apply_pooling: 
            xs = [global_mean_pool(features, batch)]
        features = F.dropout(features, p=dropout, training=self.training)
        if self.apply_pooling:
            features, edge_index, edge_attr, batch = self.do_pooling(self.pooling_1, x=features, edge_index=edge_index,edge_attr=edge_attr, batch=batch) 
        if self.number_of_layers >=2:
            features = self.do_conv(convolution_2, skip_conn_2, features, edge_index,edge_attr)
            features = self.activate(features, self.activation_type)
            if self.apply_pooling: 
                xs += [global_mean_pool(features, batch)]
            features = F.dropout(features, p=dropout, training=self.training)
            if self.apply_pooling:
                features, edge_index, edge_attr, batch = self.do_pooling(self.pooling_2, x=features, edge_index=edge_index,edge_attr=edge_attr, batch=batch) 
        if self.number_of_layers >=3:
            features = self.do_conv(convolution_3, skip_conn_3, features, edge_index,edge_attr)
            features = self.activate(features, self.activation_type)
            if self.apply_pooling: 
                xs += [global_mean_pool(features, batch)]
                features, edge_index, edge_attr, batch = self.do_pooling(self.pooling_3, x=features, edge_index=edge_index,edge_attr=edge_attr, batch=batch) 
        if self.apply_pooling:     
            features = self.jump(xs)
            features = F.relu(self.pool_lin1(features))  
        return features

    def do_pooling(self, pooling, x, edge_index, edge_attr=None, batch=None):
        if self.read_out_type =="asa_pooling":
            if self.include_edge:
                x, edge_index, edge_attr, edge_weight, batch, perm, score = pooling(x=x, edge_index=edge_index,batch=batch)
            else:
                x, edge_index, edge_weight, batch, perm = pooling(x=x, edge_index=edge_index,batch=batch)
        elif self.read_out_type =="edge_pooling":
            x, edge_index, batch, unpool_info = pooling(x=x, edge_index=edge_index,batch=batch)
        elif self.read_out_type =="sag_pooling":
            x, edge_index, edge_attr, batch, perm, score = pooling(x=x, edge_index=edge_index,edge_attr=edge_attr, batch=batch)      
        elif self.read_out_type =="topk_pooling":  
            x, edge_index, edge_attr, batch, perm, score = pooling(x=x, edge_index=edge_index,edge_attr=edge_attr, batch=batch)
        return x, edge_index, edge_attr, batch
        
             

    def do_conv_pass_without_edge(self, convolution_1, convolution_2, convolution_3, skip_conn_1, skip_conn_2, skip_conn_3, edge_index, features, dropout, batch):
        features = self.do_conv(convolution_1, skip_conn_1, features, edge_index)
        features = self.activate(features, self.activation_type)
        if self.apply_pooling: 
            xs = [global_mean_pool(features, batch)]
        features = F.dropout(features, p=dropout, training=self.training)
        if self.apply_pooling: 
            features, edge_index, edge_attr, batch = self.do_pooling(self.pooling_1, x=features, edge_index=edge_index, batch=batch)
        if self.number_of_layers >=2:
            features = self.do_conv(convolution_2, skip_conn_2, features, edge_index)
            features = self.activate(features, self.activation_type)
            if self.apply_pooling: 
                xs += [global_mean_pool(features, batch)]
            features = F.dropout(features, p=dropout, training=self.training)
            if self.apply_pooling: 
                features, edge_index, edge_attr, batch = self.do_pooling(self.pooling_2, x=features, edge_index=edge_index, batch=batch)
        if self.number_of_layers >=3:
            features = self.do_conv(convolution_3, skip_conn_3, features, edge_index)
            features = self.activate(features, self.activation_type)
            if self.apply_pooling: 
                xs += [global_mean_pool(features, batch)]
                features, edge_index, edge_attr, batch = self.do_pooling(self.pooling_3, x=features, edge_index=edge_index, batch=batch)       
        if self.apply_pooling:     
            features = self.jump(xs)
            features = F.relu(self.pool_lin1(features))
        return features

    def activate(self, features, activation_name):
        if activation_name == "relu":
            return F.relu(features)
        elif activation_name == "leaky_relu":
            return F.leaky_relu(features)
        elif activation_name == "prelu":
            return F.prelu(features, weight=0.25)    
    
    def concat_embeddings(self, embedding_s, embedding_t, include_abs=True):
        if self.include_abs:
            uv_abs = torch.abs(torch.sub(embedding_s, embedding_t))  # produces |u-v| tensor
            scores = torch.cat([embedding_s, embedding_t, uv_abs], dim=-1) # then we concatenate
        else:
            scores = torch.cat([embedding_s, embedding_t], dim=-1) 
        return scores

    def simple_read_out(self, embedding_1, x_s_batch, embedding_2, x_t_batch, size):        
        if self.read_out_type =="mean":
            embedding_1 = scatter_mean(embedding_1, x_s_batch, dim=0, dim_size=size)
            embedding_2 = scatter_mean(embedding_2, x_t_batch, dim=0, dim_size=size)
        elif self.read_out_type =="sum":
            embedding_1 = scatter_add(embedding_1, x_s_batch, dim=0, dim_size=size)
            embedding_2 = scatter_add(embedding_2, x_t_batch, dim=0, dim_size=size)     
        elif self.apply_pooling:
            pass
        return embedding_1, embedding_2
    
    def scoring_pass(self, scores):
        score = None 
        if self.scoring_type == '3layers':
            scores = self.activate(self.fully_connected_first(scores), self.activation_type)
            scores = self.activate(self.fully_connected_second(scores), self.activation_type)
            scores = self.activate(self.fully_connected_third(scores), self.activation_type)
            score = self.scoring_layer(scores).view(-1)
        elif self.scoring_type == 'simple_activation_batch':
            score = self.scoring(scores).view(-1)
        return score