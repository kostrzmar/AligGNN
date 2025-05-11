from abc import ABC, abstractmethod
import importlib
from utils import config_const
import torch.nn.functional as F


class AlignmentFactory(ABC):
    
    def get_alignment_model(self, params):
        mod = importlib.import_module(params[config_const.CONF_ALIGNMENT_MODEL])
        return mod.get_model(params)


