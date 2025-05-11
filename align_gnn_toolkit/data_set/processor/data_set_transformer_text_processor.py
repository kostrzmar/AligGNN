from data_set.processor.data_set_text_processor_abstract import DataSetTextProcessor
from transformers import AutoModel, AutoTokenizer
from torchtext.vocab import vocab
from collections import OrderedDict
import torch
from torch_scatter import scatter_mean
from typing import List, Tuple



class DataSetTransformerTextProcessor(DataSetTextProcessor):
    
    def __init__(self, 
                params=None,
                ) -> None:
        self.transformer_model = None
        self.transformers_fp16=False
        self.tokenizer_for_splitted_word = None
        super().__init__(params)   
    
    UNK_WORD = "[UNK]"


    
    def get_vocab_embedding_size(self):
        if not self.transformer_model:
            self.initialize_transformer_model()
        self.vocab_embedding_size = self.transformer_model.embeddings.word_embeddings.embedding_dim
        if isinstance(self.vocab_embedding_size, torch.Size):
            self.vocab_embedding_size = self.vocab_embedding_size[0] 
        return self.vocab_embedding_size
    

                    
    def initialize_transformer_model(self):
        if not self.transformer_model:
            embedding_transformer_model = self.params["vector.embedding.transformer_model"]
            self.transformer_model = AutoModel.from_pretrained(embedding_transformer_model, output_hidden_states=True).to(self.device)
            self.transformer_model.eval()
            if self.transformers_fp16:
                self.transformer_model = self.transformer_model.half()
                
    def initialize_transformer_tokenizer(self, is_per_words=False):
        embedding_transformer_model = self.params["vector.embedding.transformer_model"]
        if is_per_words:
            self.tokenizer_for_splitted_word = AutoTokenizer.from_pretrained(embedding_transformer_model, use_fast=True, add_prefix_space=True, add_special_tokens=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(embedding_transformer_model, use_fast=True, add_special_tokens=False) 
    
    def get_def_tokenizer(self, sentence, is_per_words=False):
        if is_per_words:
            if not self.tokenizer_for_splitted_word:
                self.initialize_transformer_tokenizer(is_per_words=is_per_words)
            return self.tokenizer_for_splitted_word.tokenize(sentence, is_split_into_words=True)
        else:
            if not self.tokenizer:
                self.initialize_transformer_tokenizer(is_per_words=is_per_words)
            return self.tokenizer.tokenize(sentence)
    
    def build_vocab(self, tokens):
        self._build_vocab_transformer(tokens,  None, None, self.get_def_unk_word_str())  
        
    def build_vocab_embeddings(self, tokens):
        if  not self.vocab_embedding:
            self.vocab_embedding = torch.zeros(1, self.get_vocab_embedding_size())
        
    def get_def_unk_word_str(self):
        if not self.tokenizer:
            self.initialize_transformer_tokenizer() 
        return self.tokenizer.unk_token


    def _build_vocab_transformer(self, tokens,  min_freq, special_tokens, default_index_token):
        if not self.tokenizer:
            self.initialize_transformer_tokenizer()
        self.vocab = vocab(OrderedDict(sorted(self.tokenizer.vocab.items(), key=lambda t: t[1])), min_freq=0)
        self.vocab.set_default_index(self.vocab[default_index_token])
        
    def is_embeddings_case_sensitive(self):
        return True   
    
    def is_sentence_ids_from_graph(self):
        return True     
    
    def get_sentence_as_text(self, sentence):
        if not self.tokenizer:
            self.initialize_transformer_tokenizer()
        return self.tokenizer.decode(sentence)
    
    def get_tokenizer_indices(self, sentence_tokens_for_graph, extract_words=True):
        indices = []
        if not self.tokenizer_for_splitted_word:
            self.initialize_transformer_tokenizer(is_per_words=True)
        words = None
        if extract_words:
            words = [word["word"] for word in sentence_tokens_for_graph]
        else:
            words = [word for word in sentence_tokens_for_graph]
        indices = self.tokenizer_for_splitted_word(words, return_tensors='pt', padding=True, is_split_into_words=True).word_ids(0)[1:-1]
        
        return indices


  
    def get_def_word_embedding(self, sentence_token_ids, sentence_tokens_for_graph: List[str], extract_words=True):
        out = {}
        indices = self.get_tokenizer_indices(sentence_tokens_for_graph, extract_words)
        if len(indices) > len(sentence_token_ids):
            self.align_tokenizations(sentence_token_ids, sentence_tokens_for_graph, indices)
        elif len(indices) < len(sentence_token_ids):
            print("ERROR graph tokenization lower then from transformer")
        with torch.no_grad():                
            tokens_tensor,segments_tensors = self.convert_sentence_to_transformer(sentence_token_ids)
            if not self.transformer_model:
                self.initialize_transformer_model()
            transformer_output: Tuple[torch.Tensor] = self.transformer_model(tokens_tensor.to(self.device), segments_tensors.to(self.device))["hidden_states"]  
            layers_to_pool = None
            pooling_dimension: int = 4 
            pooling_last_strategy=True
            if pooling_last_strategy:
                layers_to_pool = transformer_output[-pooling_dimension:]
            else:
                layers_to_pool = transformer_output[:-pooling_dimension]
            transformer_output = torch.mean(torch.stack(layers_to_pool, dim=-1), dim=-1)[:, 1:-1, :]  
            emb_per_graph_node = scatter_mean(transformer_output, index=torch.LongTensor(indices).to(self.device), dim=1).squeeze(0).cpu()
            for index in range(len(sentence_tokens_for_graph)):
                out[index] = emb_per_graph_node[index]
            
        return out,indices, transformer_output.squeeze(0).cpu()
