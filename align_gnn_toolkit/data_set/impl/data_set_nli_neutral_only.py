from datasets import load_dataset, concatenate_datasets

from data_set.data_set_abstract import DatasetAbstract

class NLIDataset(DatasetAbstract):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, type="train", params=None, graph_builder=None, data_set_processor=None):  
        DATA_SET_NAME = "snli"
        self.include_mnli = True
        self.DATA_SET_NAME_2 = "glue"
        self.DATA_SET_NAME_2_TYPE = "mnli"
        super().__init__(root, transform, pre_transform, pre_filter, data_set=DATA_SET_NAME, type=type, params=params, graph_builder=graph_builder, data_set_processor=data_set_processor)        
    
    def initialize_data_set_parameters(self):
        self.corpus_score_normalization_const = None
        self.corpus_score_normalization_min = None
        self.corpus_score_normalization_max = None
        self.corpus_attribute_name = {  NLIDataset.SOURCE+ NLIDataset.SENTENCE_TOKEN_NAME:"premise",   
                                        NLIDataset.TARGET+ NLIDataset.SENTENCE_TOKEN_NAME:"hypothesis",  
                                        NLIDataset.SCORE_TOKEN_NAME:"label"
        }   
        self.data_set_type_name = {NLIDataset.DATA_SET_TYPE_TRAIN:"train", 
                                   NLIDataset.DATA_SET_TYPE_VALIDATION:"validation", 
                                   NLIDataset.DATA_SET_TYPE_TEST:"test"  
        }

    def load_dataset_by_type(self, type, limit=None):
        dataset = None
        _type, _limit = None, None
        def num_there(s):
            return any(i.isdigit() for i in s) 
        if num_there(type):
            _type = type.split('[')[0]
            _limit = "".join([s for s in type.split('[')[1] if s.isdigit() or s.isspace()])
        else:
            _type = type
                
        snli = load_dataset(self.data_set, split=_type)
        if  self.include_mnli :   
            if _type == self.data_set_type_name[NLIDataset.DATA_SET_TYPE_VALIDATION]:
                _type = 'validation_matched'
            elif _type == self.data_set_type_name[NLIDataset.DATA_SET_TYPE_TEST]:
                _type = 'test_matched'
          
            
            mnli = load_dataset(self.DATA_SET_NAME_2, self.DATA_SET_NAME_2_TYPE, split=_type)
            mnli = mnli.remove_columns(['idx'])
            snli = snli.cast(mnli.features)
            dataset = concatenate_datasets([snli, mnli]) 
            del snli, mnli
        else:
            dataset = concatenate_datasets([snli])
            del snli

        dataset = dataset.filter(
            lambda x: True if x['label'] == 0 else False
        )
        dataset = dataset.map(lambda x: {"label": 1} , batched=False)
        if _limit:
                return dataset.select(range(int(_limit)))
        return dataset

