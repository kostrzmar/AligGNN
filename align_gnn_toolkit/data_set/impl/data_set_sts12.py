from datasets import load_dataset
from data_set.data_set_abstract import DatasetAbstract

class Sts12Dataset(DatasetAbstract):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, type="train", params=None, graph_builder=None, data_set_processor=None):  
        DATA_SET_NAME = "mteb/sts12-sts"
        super().__init__(root, transform, pre_transform, pre_filter, data_set=DATA_SET_NAME, type=type, params=params, graph_builder=graph_builder, data_set_processor=data_set_processor)        
    
    def initialize_data_set_parameters(self):
        self.corpus_score_normalization_const = None
        self.corpus_score_normalization_min = 0
        self.corpus_score_normalization_max = 5
        self.corpus_attribute_name = {  Sts12Dataset.SOURCE+ Sts12Dataset.SENTENCE_TOKEN_NAME:"sentence1",   
                                        Sts12Dataset.TARGET+ Sts12Dataset.SENTENCE_TOKEN_NAME:"sentence2",  
                                        Sts12Dataset.SCORE_TOKEN_NAME:"score"
        }   
        self.data_set_type_name = {Sts12Dataset.DATA_SET_TYPE_TRAIN:"train", 
                                   Sts12Dataset.DATA_SET_TYPE_VALIDATION:None, 
                                   Sts12Dataset.DATA_SET_TYPE_TEST:"test"  
        }

    def load_dataset_by_type(self, type, limit=None):
        return load_dataset(self.data_set, split=type)
    
