from datasets import Dataset
from data_set.data_set_abstract import DatasetAbstract

class CustomDataset(DatasetAbstract):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, type="train", params=None, graph_builder=None, data_set_processor=None):
        DATA_SET_NAME = "custom_data_set"  
        super().__init__(root, transform, pre_transform, pre_filter, data_set=DATA_SET_NAME, type=type, params=params, graph_builder=graph_builder, data_set_processor=data_set_processor)        


    def initialize_data_set_parameters(self):
        self.corpus_score_normalization_const = 3
        self.corpus_score_labels = {"notAligned":0,
                                    "aligned":1,
                                    "partialAligned":0.8
        }
        self.corpus_attribute_name = {  CustomDataset.SOURCE+CustomDataset.SENTENCE_TOKEN_NAME:"src_text",   
                                        CustomDataset.TARGET+CustomDataset.SENTENCE_TOKEN_NAME:"trg_text",  
                                        CustomDataset.SCORE_TOKEN_NAME:"aligment"
        }   
        

        self.data_set_type_name = {CustomDataset.DATA_SET_TYPE_TRAIN:"train", 
                                CustomDataset.DATA_SET_TYPE_VALIDATION:"validation", 
                                CustomDataset.DATA_SET_TYPE_TEST:"test"
        }
        

    def load_dataset_by_type(self, type, limit=None):
        return self._load_data_set(type, limit)
    
    def _load_data_set(self, data_set_type, limit):
        org = self.params["data.custom_dataset_org"]
        trg = self.params["data.custom_dataset_trg"]
        scores = self.params["data.custom_dataset_scores"]
        labels = self.params["data.custom_dataset_labels"]
        dataset = Dataset.from_dict({labels['org']: org, labels['trg']: trg, labels['scr']: scores})
        
        return dataset
