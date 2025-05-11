
from torchtext.data.utils import get_tokenizer
from collections import OrderedDict
from torchtext.vocab import build_vocab_from_iterator
from sentence_transformers import SentenceTransformer
from abc import abstractmethod 
from typing import List, Tuple
import os
import torch


class DataSetTextProcessor():
    def __init__(self, 
                params=None,
                ) -> None:
        self.params = params
        self.vocab = None
        self.vocab_embedding = None
        self.tokenizer = None
        self.vocab_embedding_size = None
        self.sentence_transformer_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @abstractmethod    
    def build_vocab(self, tokens): 
        pass   
    
    @abstractmethod  
    def get_vocab_embedding_size(self):
        pass
    
    @abstractmethod
    def get_def_tokenizer(self, sentence, is_per_words=False):
        pass
    
    @abstractmethod
    def build_vocab_embeddings(self, tokens):
        pass
    
    @abstractmethod
    def get_def_unk_word_str(self):
        pass
    
    @abstractmethod
    def get_def_word_embedding(self, sentence_token_ids, sentence_tokens_for_graph: List[str]):
        pass
    
    @abstractmethod
    def get_tokenizer_indices(self, sentence_tokens_for_graph, extract_words=True):
        pass
    
    
    def get_sentence_transfomer_embedding_size(self):
        if not self.sentence_transformer_model:
            self.initialize_sentence_transformer_model()
        if self.sentence_transformer_model:
            return self.sentence_transformer_model.get_sentence_embedding_dimension() 
        return None        
    
    def initialize_sentence_transformer_model(self):
        if not  self.sentence_transformer_model and "embedding.sentence_transformer_model" in self.params  and self.params["embedding.sentence_transformer_model"] is not None:
            embedding_transformer_model = self.params["embedding.sentence_transformer_model"]
            self.sentence_transformer_model = SentenceTransformer(embedding_transformer_model)    
    
    def get_sentence_embedding(self, sentence_as_text):
        if not self.sentence_transformer_model:
            self.initialize_sentence_transformer_model()
        return self.sentence_transformer_model.encode(sentence_as_text, convert_to_tensor=True)
    
        
    def is_embeddings_case_sensitive(self):
        return False      
    
    def is_sentence_ids_from_graph(self):
        return False      
    
    def get_embedding_language(self):
        embedding_lang = "en"
        if "vector.embedding_lang" in self.params:
            embedding_lang = self.params["vector.embedding_lang"]
        return embedding_lang
    
    def initialize_spacy_tokenizer(self, embedding_lang):
        spacy_model = "en_core_web_lg"
        if embedding_lang == 'de':
            spacy_model = "de_core_news_lg"
        self.tokenizer  = get_tokenizer("spacy", language=spacy_model)
        
    def get_token_to_index(self, token):
        return self.vocab.get_stoi()[token]
    
    
    def _build_vocab(self, tokens,  min_freq, special_tokens, default_index_token):
        def yield_tokens(data_iter):
            for text in data_iter:
                yield [x.lower() for x in self.get_def_tokenizer(text)]       
        
        self.vocab = build_vocab_from_iterator(yield_tokens(tokens),
                                        min_freq=min_freq,
                                        specials=special_tokens)
        self.vocab.set_default_index(self.vocab[default_index_token])        
    

    def build_vocables(self, from_sets, from_columns):
        def get_all_sentence(sets, columns):
            out = []
            for column in columns:
                out.extend(sets[column])
            return out
        out = get_all_sentence(from_sets, from_columns)
        self.build_vocab(out)
        self.build_vocab_embeddings(out)
        
    
    def get_def_unk_word(self):
        return self.vocab[self.get_def_unk_word_str()]
    
    
    
    # https://github.com/huggingface/transformers/issues/12665    
    def align_tokenizations(self, sentence_token_ids, sentence_tokens_for_graph, indices):
        a = []
        b = []
        def find_match(a,x,start_index):
            
            for j in range(start_index, min(start_index+forward_scan,len(a))):
                if a[j] ==x:
                    return j
            return -1
        
        def get_gaps(founded, not_founded):
            gaps = {}
            for i in range(len(founded)-1):
                if founded[i+1] -founded[i]>1:
                    gaps[founded[i]+1] = founded[i+1] -founded[i] -1
            return gaps

        for i in sentence_token_ids:
            a.append(self.vocab.get_itos()[i])
        for idx_word, word in enumerate(sentence_tokens_for_graph):
            word_tokenized = self.tokenizer.tokenize(word["word"])
            b.extend(word_tokenized)
        founded = []
        not_founded = []
        last_found_id = -1
        forward_scan = 4

        for i,x in enumerate(b):
            if i < len(a) or last_found_id < len(a):
                if i < len(a):
                    if a[i]==x:
                        founded.append(i)
                        last_found_id = i
                        continue
                    else:
                        if last_found_id < len(a):
                            index = find_match(a,x,last_found_id)
                            if index <0:
                                not_founded.append(i)
                            else:
                                founded.append(i)
                                last_found_id = index
                else:
                    if last_found_id < len(a):
                        index = find_match(a,x,last_found_id)
                        if index <0:
                            not_founded.append(i)
                        else:
                            founded.append(i)
                            last_found_id = index


        gaps = get_gaps(founded, not_founded)
        verbose = True
        if verbose:
            print(a)
            print(b)
            
        for key in gaps.keys():
            if indices[key+1] == indices[key+2]:
                del indices[key+1:key+gaps[key]]
                if verbose:
                    print(f'Remove indices between {key+1} and {key+gaps[key]}') 
            else:
                sentence_token_ids.insert(key+1, self.get_def_unk_word())
                if verbose:
                    print(f'Add into bert tokens {key+1}')
        if verbose:
            print("-"*len(b))

    def convert_sentence_to_transformer(self, sentence_token_ids):
        cls_voc_id = self.vocab.get_stoi()[self.tokenizer.cls_token]
        sep_voc_id = self.vocab.get_stoi()[self.tokenizer.sep_token]
        local_sentence = sentence_token_ids.copy()
        if cls_voc_id != sentence_token_ids[0]:
            local_sentence.insert(0, cls_voc_id)
        if sep_voc_id != sentence_token_ids[-1]:
            local_sentence.append(sep_voc_id)  
        tokens_tensor = torch.tensor(local_sentence).view(1,-1)
        segments_tensors = torch.tensor([1]*len(local_sentence)).view(1,-1)
        return tokens_tensor, segments_tensors        

    def get_cache_folderI(self):
        if "dataset.path_to_root" in self.params:
            return os.path.join(self.params["dataset.path_to_root"], ".vector_cache")
        return None