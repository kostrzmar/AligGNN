from datasets import load_dataset
from data_set.data_set_abstract import DatasetAbstract
import os
       

class GermanEval2022DatasetTest(DatasetAbstract):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, type="train", params=None, graph_builder=None, data_set_processor=None):
        DATA_SET_NAME = "german_eval_2022_test"   
        super().__init__(root, transform, pre_transform, pre_filter, data_set=DATA_SET_NAME, type=type, params=params, graph_builder=graph_builder, data_set_processor=data_set_processor)        

    
    def initialize_data_set_parameters(self):
        self.corpus_score_normalization_const = None
        self.corpus_score_normalization_min = 1
        self.corpus_score_normalization_max = 7
        
        
        
        self.corpus_attribute_name = {  GermanEval2022DatasetTest.SOURCE+GermanEval2022DatasetTest.SENTENCE_TOKEN_NAME:"Sentence DE",   
                                        GermanEval2022DatasetTest.TARGET+GermanEval2022DatasetTest.SENTENCE_TOKEN_NAME:"Sentence DE",  
                                        GermanEval2022DatasetTest.SCORE_TOKEN_NAME:"MOS"
        }   
        self.data_set_type_name = {GermanEval2022DatasetTest.DATA_SET_TYPE_TRAIN:"train", 
                                   GermanEval2022DatasetTest.DATA_SET_TYPE_VALIDATION:None, 
                                   GermanEval2022DatasetTest.DATA_SET_TYPE_TEST:"test"  
        }


        self.path_to_data = self.params["german_eval_2022_test"]
                
        self.type_to_file = {self.data_set_type_name[GermanEval2022DatasetTest.DATA_SET_TYPE_TRAIN]:"training_set.csv",   
                             self.data_set_type_name[GermanEval2022DatasetTest.DATA_SET_TYPE_TEST]:"test_set.csv",
                             self.data_set_type_name[GermanEval2022DatasetTest.DATA_SET_TYPE_VALIDATION]:"valid_set.csv"}      
    
    
    def load_dataset_by_type(self, type, limit=None):
        return self._load_data_set(type, limit)
    
    def _load_data_set(self, data_set_type, limit):
        file_name  = None
        if limit:
            file_name =  self.type_to_file[data_set_type.split('[')[0]]
            return load_dataset("csv",
                skiprows=1,
                quoting=1, 
                quotechar='"',
                column_names=['ID', 'Sentence DE', 'MOS', 'Sentence EN'],  
                delimiter=',', 
                data_files=os.path.join(self.path_to_data, file_name),
                split=f'train[:{limit}]')
        else:
            file_name =  self.type_to_file[data_set_type]
            return load_dataset("csv",
                            skiprows=1,
                            quoting=1, 
                            quotechar='"',
                            column_names=['ID', 'Sentence DE', 'MOS', 'Sentence EN'],  
                            delimiter=',', 
                            data_files=os.path.join(self.path_to_data, file_name),
                            split="train"
                            )
    
    