from datasets import load_dataset, concatenate_datasets
import os
from data_set.data_set_abstract import DatasetAbstract

class XNLIDataset(DatasetAbstract):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, type="train", params=None, graph_builder=None, data_set_processor=None):  
        DATA_SET_NAME = "xnli"
        super().__init__(root, transform, pre_transform, pre_filter, data_set=DATA_SET_NAME, type=type, params=params, graph_builder=graph_builder, data_set_processor=data_set_processor)        
    
    def initialize_data_set_parameters(self):
        self.corpus_score_normalization_const = None
        self.corpus_score_normalization_min = None
        self.corpus_score_normalization_max = None
        self.corpus_attribute_name = {  XNLIDataset.SOURCE+ XNLIDataset.SENTENCE_TOKEN_NAME:"sentence1",   
                                        XNLIDataset.TARGET+ XNLIDataset.SENTENCE_TOKEN_NAME:"sentence2",  
                                        XNLIDataset.SCORE_TOKEN_NAME:"gold_label"
        }   
        self.data_set_type_name = {XNLIDataset.DATA_SET_TYPE_TRAIN:"train", 
                                   XNLIDataset.DATA_SET_TYPE_VALIDATION:"dev", 
                                   XNLIDataset.DATA_SET_TYPE_TEST:"test"  
        }
        self.path_to_data = self.params["xnli"]
        self.type_to_file = {self.data_set_type_name[XNLIDataset.DATA_SET_TYPE_TRAIN]:"xnli.test.tsv",   
                             self.data_set_type_name[XNLIDataset.DATA_SET_TYPE_TEST]:"xnli.dev.tsv",  
                             self.data_set_type_name[XNLIDataset.DATA_SET_TYPE_VALIDATION]:"xnli.dev.tsv"}  
        
        self.path_to_data_dsnli = self.params["dsnli"]
        self.type_to_file_dsnli = {self.data_set_type_name[XNLIDataset.DATA_SET_TYPE_TRAIN]:"snli_1.0_train.csv",   
                             self.data_set_type_name[XNLIDataset.DATA_SET_TYPE_TEST]:"snli_1.0_test.csv",  
                             self.data_set_type_name[XNLIDataset.DATA_SET_TYPE_VALIDATION]:"snli_1.0_dev.csv"}          

        
    def load_dataset_by_type(self, type, limit=None):
        dataset = None
        include_dsnli = True
        _type, _limit = None, None
        def num_there(s):
            return any(i.isdigit() for i in s) 
        if num_there(type):
            _type = type.split('[')[0]
            _limit = "".join([s for s in type.split('[')[1] if s.isdigit() or s.isspace()])
        else:
            _type = type
        file_name =  self.type_to_file[_type]        
        dataset =  load_dataset("csv",
                quoting=3, 
                column_names=["language","gold_label","sentence1_binary_parse","sentence2_binary_parse","sentence1_parse","sentence2_parse","sentence1","sentence2","promptID","pairID","genre","label1","label2","label3","label4","label5","sentence1_tokenized","sentence2_tokenized","match"],  
                delimiter='\t', 
                data_files=os.path.join(self.path_to_data, file_name))['train']        
 
        dataset = dataset.remove_columns(["sentence1_binary_parse","sentence2_binary_parse","sentence1_parse","sentence2_parse", "promptID","pairID","genre","label1","label2","label3","label4","label5","sentence1_tokenized","sentence2_tokenized","match"])
        dataset = dataset.filter(
            lambda x: True if x['gold_label'] == 'entailment' and x['language'] =='de' else False
        )
        dataset = dataset.map(lambda x: {"gold_label": 1} , batched=False)
        dataset = dataset.remove_columns(["language"])
        if include_dsnli:
            file_name =  self.type_to_file_dsnli[_type]        
            dataset_dsnli =  load_dataset("csv",
                    quoting=0, 
                    column_names=["sentence1","sentence2","gold_label"],  
                    delimiter=',', 
                    quotechar='"',
                    data_files=os.path.join(self.path_to_data_dsnli, file_name))['train']
            dataset_dsnli = dataset_dsnli.filter(
                lambda x: True if x['gold_label'] == 'entailment' and x['sentence2'] not in (None, "n/a", "N/A") else False
            )
            dataset_dsnli = dataset_dsnli.map(lambda x: {"gold_label": 1} , batched=False)    
            dataset = concatenate_datasets([dataset, dataset_dsnli])
            del dataset_dsnli 
        if _limit:
                return dataset.select(range(int(_limit)))
        return dataset
