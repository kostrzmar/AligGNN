from data_set.processor.data_set_text_processor_abstract import DataSetTextProcessor
from torchtext.vocab import  GloVe
from typing import List
import torch

class DataSetGloveTextProcessor(DataSetTextProcessor):
    
    def __init__(self, 
                params=None,
                ) -> None:
        self.gloveText_vector = None
        super().__init__(params)   
  
    UNK_WORD = "<unk>"
    PAD_WORD = "<pad>"
    
    def initialize_embedding(self):
        self.gloveText_vector = GloVe(name="6B", dim=300, cache=self.get_cache_folderI())
    
    def get_vocab_embedding_size(self):
        if not self.vocab_embedding_size:
            if not self.gloveText_vector:
                self.initialize_embedding()
            self.vocab_embedding_size = self.gloveText_vector.dim
        if isinstance(self.vocab_embedding_size, torch.Size):
            self.vocab_embedding_size = self.vocab_embedding_size[0] 
        return self.vocab_embedding_size
        
    def get_def_tokenizer(self, sentence, is_per_words=False):
        if not self.tokenizer:
            self.initialize_spacy_tokenizer(self.get_embedding_language())
        if is_per_words:
            return [word for word in map(lambda x: x.text,self.tokenizer.keywords['spacy'].tokenizer.pipe(sentence))]
        return self.tokenizer(sentence)
    
    def build_vocab(self, tokens):
        self._build_vocab(tokens, 1, [DataSetGloveTextProcessor.UNK_WORD, DataSetGloveTextProcessor.PAD_WORD], DataSetGloveTextProcessor.UNK_WORD)
    
    def get_tokenizer_indices(self, sentence_tokens_for_graph, extract_words=True):
        return [i for i in range(0, len(sentence_tokens_for_graph))]      
        
    def build_vocab_embeddings(self, tokens):
        if not self.gloveText_vector:
            self.initialize_embedding()
        self.vocab_embedding = self.gloveText_vector.get_vecs_by_tokens(self.vocab.get_itos())
        self.vocab_embedding_size = self.vocab_embedding[0].shape
        
    def get_def_unk_word_str(self):
            return DataSetGloveTextProcessor.UNK_WORD
        
    def get_sentence_as_text(self, sentence):
        return  " ".join([self.vocab.get_itos()[x] for x in sentence])

    def get_def_word_embedding(self, sentence_token_ids, sentence_tokens_for_graph: List[str], extract_words=True):
        out = {}
        indices = list(range(0,len(sentence_token_ids)))
        for index in range(len(sentence_token_ids)):
            out[index] = self.vocab_embedding[sentence_token_ids[index]]
        return out,indices, None
    
    def is_sentence_ids_from_graph(self):
        return True    
    