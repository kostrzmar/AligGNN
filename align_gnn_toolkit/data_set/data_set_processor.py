from data_set.processor.data_set_glove_text_processor import DataSetGloveTextProcessor 
from data_set.processor.data_set_fasttext_text_processor import DataSetFasttextTextProcessor 
from data_set.processor.data_set_bert_text_processor import DataSetBertTextProcessor 
from data_set.processor.data_set_transformer_text_processor import DataSetTransformerTextProcessor 

class DataSetProcessor():
    TEXT_PROCESSOR  = "text_processor"
    def __init__(self, 
                params=None,
                ) -> None:
        self.params = params
        self.text_processor = self.get_text_processor(param = params)
        
    def initializeProcessors(self, data_holder):
        pass
    
    def get_text_processor(self, param):
        embedding_name = self.get_embedding_name()
        if "fasttext" == embedding_name:
            return DataSetFasttextTextProcessor(params = param)
        elif "glove" == embedding_name:
            return DataSetGloveTextProcessor(params = param)
        elif "bert"  == embedding_name:    
            return DataSetBertTextProcessor(params = param)
        elif "transformer"  == embedding_name:  
            return DataSetTransformerTextProcessor(params = param)  
        
    
    def get_embedding_name(self):
        return self.params["vector.embedding_name"]    

    def getProcessor(self, processor_type=None):
        if processor_type == self.TEXT_PROCESSOR:
            return self.text_processor
        return self.text_processor