from datasets import load_dataset
from data_set.data_set_abstract import DatasetAbstract
 

class SickDataset(DatasetAbstract):

    MIN_VALUE = 1
    MAX_VALUE = 5
    
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, type="train", params=None, graph_builder=None, data_set_processor=None):
        DATA_SET_NAME = "sick"
        super().__init__(root, transform, pre_transform, pre_filter, data_set=DATA_SET_NAME, type=type, params=params, graph_builder=graph_builder, data_set_processor=data_set_processor)   
         
    def initialize_data_set_parameters(self):
        self.corpus_score_normalization_const = None
        self.corpus_score_normalization_min = SickDataset.MIN_VALUE
        self.corpus_score_normalization_max = SickDataset.MAX_VALUE
        self.corpus_attribute_name = {  SickDataset.SOURCE+SickDataset.SENTENCE_TOKEN_NAME:"sentence_A",   
                                        SickDataset.TARGET+SickDataset.SENTENCE_TOKEN_NAME:"sentence_B",  
                                        SickDataset.SCORE_TOKEN_NAME:"relatedness_score"
        }   
        self.data_set_type_name = {SickDataset.DATA_SET_TYPE_TRAIN:"train", 
                                   SickDataset.DATA_SET_TYPE_VALIDATION:"validation", 
                                   SickDataset.DATA_SET_TYPE_TEST:"test"  
        }
    def load_dataset_by_type(self, type, limit=None):
        return load_dataset(self.data_set, split=type)
    
    