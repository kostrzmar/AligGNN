from datasets import load_dataset
from data_set.data_set_abstract import DatasetAbstract
import os


       

class TestDataset(DatasetAbstract):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, type="train", params=None, graph_builder=None, data_set_processor=None):
        DATA_SET_NAME = "align_gnn_test"  
        super().__init__(root, transform, pre_transform, pre_filter, data_set=DATA_SET_NAME, type=type, params=params, graph_builder=graph_builder, data_set_processor=data_set_processor)        


    def initialize_data_set_parameters(self):
        self.corpus_score_normalization_const = 3
        
        self.corpus_score_labels = {"notAligned":0,
                                    "aligned":1,
                                    "partialAligned":0.8
        }
        self.corpus_attribute_name = {  TestDataset.SOURCE+TestDataset.SENTENCE_TOKEN_NAME:"src_text",   
                                        TestDataset.TARGET+TestDataset.SENTENCE_TOKEN_NAME:"trg_text",  
                                        TestDataset.SCORE_TOKEN_NAME:"aligment"
        }   
        self.data_set_type_name = {TestDataset.DATA_SET_TYPE_TRAIN:"train", 
                                   TestDataset.DATA_SET_TYPE_VALIDATION:"validation", 
                                   TestDataset.DATA_SET_TYPE_TEST:"test"  
        }
        
        self.path_to_data   = self.params["align_gnn_test"] 
        self.type_to_file = {self.data_set_type_name[TestDataset.DATA_SET_TYPE_TRAIN]:"train.tsv",   
                             self.data_set_type_name[TestDataset.DATA_SET_TYPE_TEST]:"test.tsv",  
                             self.data_set_type_name[TestDataset.DATA_SET_TYPE_VALIDATION]:"dev.tsv"}      

    def load_dataset_by_type(self, type, limit=None):
        return self._load_data_set(type, limit)
    
    def _load_data_set(self, data_set_type, limit):
        file_name  = None
        if self.limit:
            file_name =  self.type_to_file[data_set_type.split('[')[0]]
        else:
            file_name =  self.type_to_file[data_set_type]
        return load_dataset("csv",
                            quoting=3, 
                            column_names=['aligment', 'src_id', 'trg_id', 'src_text', 'trg_text'],  
                            delimiter='\t', 
                            data_files=os.path.join(self.path_to_data, file_name))['train']
