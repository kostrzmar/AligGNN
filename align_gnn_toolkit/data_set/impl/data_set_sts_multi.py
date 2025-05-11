from datasets import load_dataset
from data_set.data_set_abstract import DatasetAbstract



     

class StsMultiDataset(DatasetAbstract):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, type="train", params=None, graph_builder=None, data_set_processor=None):
        DATA_SET_NAME = "stsb_multi_mt"
        if type =="validation":
            type = "dev"
        super().__init__(root, transform, pre_transform, pre_filter, data_set=DATA_SET_NAME, type=type, params=params, graph_builder=graph_builder, data_set_processor=data_set_processor)        
    
    def initialize_data_set_parameters(self):
        self.corpus_score_normalization_const = None
        self.corpus_score_normalization_min = 0
        self.corpus_score_normalization_max = 5
        self.corpus_attribute_name = {  StsMultiDataset.SOURCE+StsMultiDataset.SENTENCE_TOKEN_NAME:"sentence1",   
                                        StsMultiDataset.TARGET+StsMultiDataset.SENTENCE_TOKEN_NAME:"sentence2",  
                                        StsMultiDataset.SCORE_TOKEN_NAME:"similarity_score"
        }   
        
         
        self.data_set_type_name = {StsMultiDataset.DATA_SET_TYPE_TRAIN:"train", 
                                   StsMultiDataset.DATA_SET_TYPE_VALIDATION:"dev", 
                                   StsMultiDataset.DATA_SET_TYPE_TEST:"test"  
        }


    def load_dataset_by_type(self, type, limit=None):
        return load_dataset(self.data_set, split=type, name=self.data_set_processor.text_processor.get_embedding_language())