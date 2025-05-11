from typing import List, Optional, Union
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

@functional_transform('binarize')
class Binarize(BaseTransform):

    def __init__(
        self,
        data_set
      ):
        self.max = data_set.MAX_VALUE
        self.min = data_set.MIN_VALUE

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.node_stores:
            if hasattr(store, 'y'):
                z = store.y
                org_score = z * (self.max  - self.min) + self.min
                binary_core = 1 if  org_score >=4 else 0
                store.y  =  torch.tensor(binary_core, dtype=store.y.dtype)
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(Convert to binary classification)'