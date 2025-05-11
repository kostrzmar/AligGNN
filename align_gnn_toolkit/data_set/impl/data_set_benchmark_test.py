from datasets import load_dataset
from data_set.data_set_abstract import DatasetAbstract
import os
       

class BenchmarkTestDataset(DatasetAbstract):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, type="train", params=None, graph_builder=None, data_set_processor=None):
        DATA_SET_NAME = "benchmark_test"   
        super().__init__(root, transform, pre_transform, pre_filter, data_set=DATA_SET_NAME, type=type, params=params, graph_builder=graph_builder, data_set_processor=data_set_processor)        

    
    def initialize_data_set_parameters(self):
        self.corpus_score_normalization_const = None
        self.corpus_score_normalization_min = None
        self.corpus_score_normalization_max = None
        
        self.corpus_score_labels = {"notAligned":0,
                                    "aligned":1,
                                    "partialAligned":1
        }
        
        
        self.corpus_attribute_name = {  BenchmarkTestDataset.SOURCE+BenchmarkTestDataset.SENTENCE_TOKEN_NAME:"src_text",   
                                        BenchmarkTestDataset.TARGET+BenchmarkTestDataset.SENTENCE_TOKEN_NAME:"trg_text",  
                                        BenchmarkTestDataset.SCORE_TOKEN_NAME:"aligment"
        }   
        self.data_set_type_name = {BenchmarkTestDataset.DATA_SET_TYPE_TRAIN:None, 
                                   BenchmarkTestDataset.DATA_SET_TYPE_VALIDATION:None, 
                                   BenchmarkTestDataset.DATA_SET_TYPE_TEST:"test"  
        }
        
        self.path_to_data = self.params["path_benchmark_test"]
                
        self.type_to_file = {self.data_set_type_name[BenchmarkTestDataset.DATA_SET_TYPE_TRAIN]:"benchmark.txt",   
                             self.data_set_type_name[BenchmarkTestDataset.DATA_SET_TYPE_TEST]:"benchmark.txt",  
                             self.data_set_type_name[BenchmarkTestDataset.DATA_SET_TYPE_VALIDATION]:"benchmark.txt"}      
    
    
    def load_dataset_by_type(self, type, limit=None):
        return self._load_data_set(type, limit)
    
    def _load_data_set(self, data_set_type, limit):
        file_name  = None
        if limit:
            file_name =  self.type_to_file[data_set_type.split('[')[0]]
            return load_dataset("csv",
                quoting=3, 
                column_names=['aligment', 'src_text', 'trg_text', 'id', 'score', 'method', 'dataset', 'error_type'],  
                delimiter='\t', 
                data_files=os.path.join(self.path_to_data, file_name),
                split=f'train[:{limit}]')
        else:
            file_name =  self.type_to_file[data_set_type]
            dataset =  load_dataset("csv",
                            quoting=3, 
                            column_names=['aligment', 'src_text', 'trg_text', 'id', 'score', 'method', 'dataset', 'error_type'],  
                            delimiter='\t', 
                            data_files=os.path.join(self.path_to_data, file_name))['train']

            #dataset = dataset.map(lambda x: {"aligment": 1, 'partialAligned':0.7, "notAligned":0} , batched=False)
            return dataset
    
    
    