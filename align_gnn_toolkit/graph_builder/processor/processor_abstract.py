from diskcache import Cache
import os
import logging
import hashlib
from abc import abstractmethod

class ProcessorAbstract():
    
    def __init__(self, 
            params=None,
            url=None,
            port=None,
            data_set=None
            ) -> None:
        self.params = params
        self.url = url
        self.port = port
        self.data_set = data_set
        self.lang=self.get_embedding_language()
        root_folder = "."
        if self.data_set.get_path_to_root_folder():
            root_folder = self.data_set.get_path_to_root_folder()   
        self.cache_folder = os.path.join(root_folder, "data/caches", self.getProcessorName(), self.data_set.data_set)
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder, exist_ok=True)
        self.cache = Cache(self.cache_folder)
        warnings = self.cache.check()
        for warning in warnings:
            logging.WARN(f'Cache [{self.cache_folder}] warning: {warning}')
            
        self.sentence_dict = None
        self.sentence_out_mapping = None
        


    def get_embedding_language(self):
        embedding_lang = "en"
        if "vector.embedding_lang" in self.params:
            embedding_lang = self.params["vector.embedding_lang"]
        return embedding_lang


    def preprocess_graph(self, dataset, sentence, sentence_mask):
        pass


    def registerAnnotatorsForBuilders(self, builder_names):
        pass

    @abstractmethod
    def get_tag_definition(self, type):
        pass
    
    def tag_to_index(self, type):
        return self.convert_tag_to_index(self.get_tag_definition(type))
    

    def tag_to_name(self, type):
        return self.convert_tag_to_name(self.get_tag_definition(type))
    
    
    @abstractmethod
    def getData(self, type):
        pass

    @abstractmethod
    def parseSentence(self, sentence, sentence_mask):
        pass

    def isCacheEnabled(self):
        if "dataset.use.cache" in self.params and self.params["dataset.use.cache"]:
            return True
        return False

    def isInCache(self, key):
        out = False
        if self.isCacheEnabled():
            with self.cache as reference:
                if key in reference:
                    out = True
        return out
        

    def getFromCache(self, key):
        out = None
        with self.cache as reference:
           if key in reference:
               out = reference.get(key)
        return out 
    
    def setToCache(self, key, value):
        if self.isCacheEnabled():
            with self.cache as reference:
                reference.set(key, value)
    
    def getHashFromSentence(self, sentence, post_fix=""):
        return hashlib.sha256(sentence.encode('utf-8')).hexdigest() +"_"+ post_fix

    @abstractmethod
    def getProcessorName(self):
        pass
    
    @abstractmethod
    def tokenize_sentence(self, sentence_as_text):
        pass
    
    @abstractmethod
    def isSupported(self, lang):
        pass
    
    def convert_tag_to_index(self, tag_def):
        if tag_def:
            return {t[0]:i for i, t in enumerate(tag_def)}
        return {}
    
    def convert_tag_to_name(self, tag_def):
        if tag_def:
            return {t[0]:t[1] for t in tag_def}
        return {}