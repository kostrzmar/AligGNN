from datasets import load_dataset
from data_set.data_set_abstract import DatasetAbstract
 

class SickBinaryDataset(DatasetAbstract):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, type="train", params=None, graph_builder=None, data_set_processor=None):
        DATA_SET_NAME = "sick"
        super().__init__(root, transform, pre_transform, pre_filter, data_set=DATA_SET_NAME, type=type, params=params, graph_builder=graph_builder, data_set_processor=data_set_processor)   
         
    def initialize_data_set_parameters(self):
        self.corpus_score_normalization_const = None
        self.corpus_score_normalization_min = None
        self.corpus_score_normalization_max = None


        self.corpus_attribute_name = {  SickBinaryDataset.SOURCE+SickBinaryDataset.SENTENCE_TOKEN_NAME:"sentence_A",   
                                        SickBinaryDataset.TARGET+SickBinaryDataset.SENTENCE_TOKEN_NAME:"sentence_B",  
                                        SickBinaryDataset.SCORE_TOKEN_NAME:"relatedness_score"
        }   
        self.data_set_type_name = {SickBinaryDataset.DATA_SET_TYPE_TRAIN:"train", 
                                   SickBinaryDataset.DATA_SET_TYPE_VALIDATION:"validation", 
                                   SickBinaryDataset.DATA_SET_TYPE_TEST:"test"  
        }
    def load_dataset_by_type(self, type, limit=None):
        return load_dataset(self.data_set, split=type)
    
    def load_dataset_by_type(self, type, limit=None):
        sick = None
        _type, _limit = None, None
        def num_there(s):
            return any(i.isdigit() for i in s) 
        if num_there(type):
            _type = type.split('[')[0]
            _limit = "".join([s for s in type.split('[')[1] if s.isdigit() or s.isspace()])
        else:
            _type = type
        sick = load_dataset(self.data_set, split=_type) 
        def binarize(item):
            item["relatedness_score"] = 1 if float(item["relatedness_score"])>=4 else 0
            return item
        sick = sick.map(binarize)
        if _limit:
                return sick.select(range(int(_limit)))
        return sick