import torch
import os.path as osp
from data_set.data_set_abstract import DatasetAbstract
class SickHeteroDataset(DatasetAbstract):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, type="train", params=None, graph_builder=None, data_set_processor=None):
        DATA_SET_NAME = "sick_hetero"
        super().__init__(root, transform, pre_transform, pre_filter, data_set=DATA_SET_NAME, type=type, params=params, graph_builder=graph_builder, data_set_processor=data_set_processor)   


             
    def initialize_data_set_parameters(self):
        self.data_set_type_name = {SickHeteroDataset.DATA_SET_TYPE_TRAIN:"train", 
                                   SickHeteroDataset.DATA_SET_TYPE_VALIDATION:"validation", 
                                   SickHeteroDataset.DATA_SET_TYPE_TEST:"test"  
        }

            
    
    def load_dataset_by_type(self, type, limit=None):
        pass
    
    def download(self):
        pass
    
    def process(self):
        pass
    
    
    def len(self):
        if self.type == 'test':
            return 4906
        if self.type == 'train':
            return 4439
        if self.type == 'validation':
            return 495


    
    def get(self, idx):
        try:
            data = torch.load(osp.join(self.processed_dir, f'data_{self.type}_{idx}.pt'))
        except RuntimeError as err:
            print(f'Error loading data_{self.type}_{idx}.pt', err)
        return data        