from datasets import load_dataset
from data_set.data_set_abstract import DatasetAbstract

class StsDataset(DatasetAbstract):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, type="train", params=None, graph_builder=None, data_set_processor=None):  
        DATA_SET_NAME = "mteb/stsbenchmark-sts"
        super().__init__(root, transform, pre_transform, pre_filter, data_set=DATA_SET_NAME, type=type, params=params, graph_builder=graph_builder, data_set_processor=data_set_processor)        
    
    def initialize_data_set_parameters(self):
        self.corpus_score_normalization_const = None
        self.corpus_score_normalization_min = 0
        self.corpus_score_normalization_max = 5
        self.corpus_attribute_name = {  StsDataset.SOURCE+ StsDataset.SENTENCE_TOKEN_NAME:"sentence1",   
                                        StsDataset.TARGET+ StsDataset.SENTENCE_TOKEN_NAME:"sentence2",  
                                        StsDataset.SCORE_TOKEN_NAME:"score"
        }   
        self.data_set_type_name = {StsDataset.DATA_SET_TYPE_TRAIN:"train", 
                                   StsDataset.DATA_SET_TYPE_VALIDATION:"validation", 
                                   StsDataset.DATA_SET_TYPE_TEST:"test"  
        }

    def load_dataset_by_type(self, type, limit=None):
        return load_dataset(self.data_set, split=type)
