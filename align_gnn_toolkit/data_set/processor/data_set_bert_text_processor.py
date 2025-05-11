from data_set.processor.data_set_text_processor_abstract import DataSetTextProcessor
from transformers import BertModel
from transformers import BertTokenizerFast
from torchtext.vocab import vocab
from collections import OrderedDict
import torch
from torch_scatter import scatter_mean
from typing import List

class DataSetBertTextProcessor(DataSetTextProcessor):
    
    def __init__(self, 
                params=None,
                ) -> None:
        self.bert_model = None
        super().__init__(params)   
    
    BERT_MODEL_EN = 'bert-base-uncased'
    BERT_MODEL_DE = 'dbmdz/bert-base-german-uncased'
    UNK_WORD = "[UNK]"

    def is_sentence_ids_from_graph(self):
        return True     
    
    def get_vocab_embedding_size(self):
        if not self.vocab_embedding_size:
            self.vocab_embedding_size = 768
        if isinstance(self.vocab_embedding_size, torch.Size):
            self.vocab_embedding_size = self.vocab_embedding_size[0] 
        return self.vocab_embedding_size
    
    def get_model_name(self, embedding_lang):
        bert_model = DataSetBertTextProcessor.BERT_MODEL_EN
        if embedding_lang == 'de':
            bert_model = DataSetBertTextProcessor.BERT_MODEL_DE
        return bert_model
            
    def init_bert_model(self):
        if not self.bert_model:
            self.bert_model = BertModel.from_pretrained(self.get_model_name(self.get_embedding_language()),output_hidden_states = True,)
        if not self.tokenizer:
            self.initialize_bert_tokenizer()
    
    def initialize_bert_tokenizer(self):
        if not self.tokenizer:
            self.tokenizer = BertTokenizerFast.from_pretrained(self.get_model_name(self.get_embedding_language()))

    def get_def_tokenizer(self, sentence, is_per_words=False):
        if not self.tokenizer:
            self.initialize_bert_tokenizer()
        if is_per_words:
            sentence = " ".join(sentence).strip()
        return self.tokenizer.tokenize(sentence)
    
    def build_vocab(self, tokens):
        self._build_vocab_bert(tokens,  None, None, DataSetBertTextProcessor.UNK_WORD)
        
        
    def build_vocab_embeddings(self, tokens):
        if  not self.vocab_embedding:
            self.vocab_embedding = torch.zeros(1, self.get_vocab_embedding_size())
        
    def get_def_unk_word_str(self):
            return DataSetBertTextProcessor.UNK_WORD
        
    def _build_vocab_bert(self, tokens,  min_freq, special_tokens, default_index_token):
        self.init_bert_model()
        self.vocab = vocab(OrderedDict(sorted(self.tokenizer.vocab.items(), key=lambda t: t[1])), min_freq=0)
        self.vocab.set_default_index(self.vocab[default_index_token])
        
        
    def get_sentence_as_text(self, sentence):
        if not self.tokenizer:
            self.initialize_bert_tokenizer()
        return self.tokenizer.decode(sentence)
    
    def bert_text_preparation(self,text):
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.get_def_tokenizer(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1]*len(indexed_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        return tokenized_text, tokens_tensor, segments_tensors

    def get_bert_embeddings(self,tokens_tensor, segments_tensors, strategy='avg_last_four', to_list = True):
        if not self.bert_model:
            self.init_bert_model()
        with torch.no_grad():
            outputs = self.bert_model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2][1:]
        
        if strategy=='last':
            token_embeddings = hidden_states[-1]
        elif strategy=='second_to_last':
            token_embeddings = hidden_states[-2]
        elif strategy=='sum_last_four':
            token_embeddings = torch.stack(hidden_states[-4:]).sum(0)
        elif strategy=='avg_last_four':
            token_embeddings = torch.stack(hidden_states[-4:]).mean(0)    
        if to_list:
            token_embeddings = torch.squeeze(token_embeddings, dim=0)
            list_token_embeddings = token_embeddings.tolist()
            return list_token_embeddings
        else:
            return token_embeddings
        
    def get_tokenizer_indices(self, sentence_tokens_for_graph, extract_words=True):
        indices = []
        for idx_word, word in enumerate(sentence_tokens_for_graph):
            _word = None
            if extract_words:
                _word = word["word"]
            else:
                _word = word
                
            word_tokenized = self.tokenizer.tokenize(_word)
            for _ in range(len(word_tokenized)):
                indices.append(idx_word)
        return indices

    def get_def_word_embedding(self, sentence_token_ids, sentence_tokens_for_graph: List[str], extract_words=True):
        out = {}
        indices = []
        indices = self.get_tokenizer_indices(sentence_tokens_for_graph, extract_words)
        if len(indices) > len(sentence_token_ids):
            self.align_tokenizations(sentence_token_ids, sentence_tokens_for_graph, indices)
        elif len(indices) < len(sentence_token_ids):
            print("ERROR graph tokenization lower then from transformer")                
        tokens_tensor,segments_tensors = self.convert_sentence_to_transformer(sentence_token_ids)
        list_token_embeddings = self.get_bert_embeddings(tokens_tensor, segments_tensors, to_list=False)
        emb_per_graph_node = scatter_mean(list_token_embeddings[:,1:-1,:], index=torch.LongTensor(indices), dim=1).squeeze(0)
        for index in range(len(sentence_tokens_for_graph)):
            out[index] = emb_per_graph_node[index]
        return out,indices, None
    
