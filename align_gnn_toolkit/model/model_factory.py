import torch
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import numpy as np
from utils import executor_utils as utils
from abc import ABC
import importlib
from utils import config_const
import torch.nn.functional as F


class ModelFactory(ABC):
    
    def get_model(self, params, in_dim, edge_dim, vocab=None) -> torch.nn.Module:
        mod = importlib.import_module(params[config_const.CONF_LEARNER_NAME])
        return mod.get_model(params, in_dim, edge_dim, vocab)

    def do_train(self, params, model, train_loader, optimizer, device):
        mod = importlib.import_module(params[config_const.CONF_LEARNER_NAME])
        return  mod.train(model, train_loader, optimizer, device)

    def do_eval(self, params, model, loader, device, type):
        mod = importlib.import_module(params[config_const.CONF_LEARNER_NAME])
        return mod.eval(model, loader, device, type)
  
    def do_predict(self, params, model, test_data_loader, device, embedding_as_numpy=True):
        mod = importlib.import_module(params[config_const.CONF_LEARNER_NAME])
        return mod.predict(model, test_data_loader, device, embedding_as_numpy)

    def compare_results(self, best_eval, current_eval, is_lower_better=True):
        if is_lower_better:
            return best_eval is None or (current_eval) < (best_eval)
        else:
            return best_eval is None or (current_eval) > (best_eval)    
        
        
    def load_model_from_path(self, model, path):
        return utils.load_model_from_path(model, path)
