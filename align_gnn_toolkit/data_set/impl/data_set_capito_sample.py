from datasets import load_dataset
from datasets import Features, Value
from data_set.data_set_abstract import DatasetAbstract
from datasets import concatenate_datasets
import os
import numpy as np
 

class CapitoSampleDataset(DatasetAbstract):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, type="train", params=None, graph_builder=None, data_set_processor=None):
        DATA_SET_NAME = "capito_sample"  
        super().__init__(root, transform, pre_transform, pre_filter, data_set=DATA_SET_NAME, type=type, params=params, graph_builder=graph_builder, data_set_processor=data_set_processor)        


    def initialize_data_set_parameters(self):
        self.corpus_score_normalization_const = 3
        

        self.corpus_attribute_name = {  CapitoSampleDataset.SOURCE_SENTENCE_TOKEN_NAME:"src_text",   
                                        CapitoSampleDataset.TARGET_SENTENCE_TOKEN_NAME:"trg_text",  
                                        CapitoSampleDataset.SCORE_TOKEN_NAME:"score"
        }   
        self.data_set_type_name = {CapitoSampleDataset.DATA_SET_TYPE_TRAIN:"train", 
                                   CapitoSampleDataset.DATA_SET_TYPE_VALIDATION:"validation", 
                                   CapitoSampleDataset.DATA_SET_TYPE_TEST:"test"  
        }
        
        self.path_to_data = self.params["capito_sample"]
        
        allFiles = sorted([f for f in os.listdir(self.path_to_data) if f.endswith(".csv")])
        train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFiles),  [int(len(allFiles)*0.7), int(len(allFiles)*0.85)])

        
        self.type_to_file = {self.data_set_type_name[CapitoSampleDataset.DATA_SET_TYPE_TRAIN]:train_FileNames,   
                             self.data_set_type_name[CapitoSampleDataset.DATA_SET_TYPE_TEST]:test_FileNames,  
                             self.data_set_type_name[CapitoSampleDataset.DATA_SET_TYPE_VALIDATION]:val_FileNames}      

    def load_dataset_by_type(self, type, limit=None):
        return self._load_data_set(type, limit)
    
    def _load_data_set(self, data_set_type, limit):
        file_names  = None
        if self.limit:
            file_names =  self.type_to_file[data_set_type.split('[')[0]]
        else:
            file_names =  self.type_to_file[data_set_type]
        
        out_ds = None
        ft = Features({'src_text': Value(dtype='string', id=None), 'trg_text': Value(dtype='string', id=None), 'score': Value(dtype='float64', id=None)})
        for file_name in file_names:
            ds = load_dataset("csv",
                            column_names=['src_text', 'trg_text', 'score'],  
                            delimiter=',', 
                            data_files=os.path.join(self.path_to_data, file_name), features=ft,  keep_default_na=False)['train']
            if out_ds:
                out_ds = concatenate_datasets([out_ds, ds])
            else: 
                out_ds = ds
            
            
        return out_ds
