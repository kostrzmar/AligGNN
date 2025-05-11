from graph_builder.graph_builder_type_abstract import GraphBuilderTypeAbstract
from data_set.data_set_processor import DataSetProcessor
from data_set.data_set_abstract import DatasetAbstract
from collections import defaultdict
from utils import config_const
class GraphBuilderSequence(GraphBuilderTypeAbstract):
    def __init__(self, 
                params=None,
                master_builder=None, 
                is_master = False
                ) -> None:
        super(GraphBuilderSequence, self).__init__(params,  "Sequence")
        

    def isNodeOHE(self):
        return False
        
    def isEdgeOHE(self):
        return True
    
    def getNodeFeatsNbr(self):
        return self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_vocab_embedding_size() 

    def getEdgeFeatsNbr(self): 
        return len(self.getEdgeLabels())

    def clean_up_token(self, token):
        return token.strip()
    
    def case_token(self, token, _text_processor):
        if _text_processor.is_embeddings_case_sensitive():
            return token
        return token.lower()

    def customize_graph(self,item, item_type, node_feats,edge_index,node_labels, edge_feats, graph_dict):
        info_for_graphs = self.getGraphData()
        text_processor = self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR)
        _unk_word_str = text_processor.get_def_unk_word_str()
        stoi = text_processor.vocab.get_stoi()
            
        for info_for_graph in info_for_graphs:
            tokens_for_nodes = info_for_graph["tokens"]
            
            if text_processor.is_sentence_ids_from_graph():                
                item[self.data_set.get_token_name(item_type,DatasetAbstract.TOKEN)] = [x if x in _unk_word_str else self.case_token(x, text_processor) for x in text_processor.get_def_tokenizer([tokens_for_nodes[i]["word"] for  i  in range(len(tokens_for_nodes))], is_per_words=True)]
            else:
                item[self.data_set.get_token_name(item_type,DatasetAbstract.TOKEN)]  = [x if x in _unk_word_str else self.case_token(x, text_processor) for x in text_processor.get_def_tokenizer(self.clean_up_token(item[self.data_set.get_token_name(item_type,DatasetAbstract.SENTENCE_TOKEN_NAME)]))]

            item[self.data_set.get_token_name(item_type,DatasetAbstract.TOKEN_VOCAB_ID)] = [stoi[x] if x in stoi else stoi[_unk_word_str] for x in item[self.data_set.get_token_name(item_type,DatasetAbstract.TOKEN)]]

            
            tokens_ids_tokenized = item[self.data_set.get_token_name(item_type,DatasetAbstract.TOKEN_VOCAB_ID)]
            emb, indices, _  = self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_def_word_embedding(tokens_ids_tokenized, tokens_for_nodes)
            item[self.data_set.get_token_name(item_type,DatasetAbstract.SENTENCE_TOKEN_VOCAB_ID_MASK)] = indices
            
            if config_const.CONF_EMBEDDING_SENTENCE_TRANSFORMER_MODEL in self.params and self.params[config_const.CONF_EMBEDDING_SENTENCE_TRANSFORMER_MODEL] is not None:

                sentence = self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_sentence_as_text(tokens_ids_tokenized)
                graph_dict["sent_embeddings"] =self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_sentence_embedding(sentence)
            
            i_to_p = defaultdict(list)
            for pos, index in enumerate(indices):
                i_to_p[index].append(pos)

            
            for index, _ in enumerate(tokens_for_nodes):
                node_feats.append(
                    self.update_node_feats(
                        self.init_node_feats(),
                                emb[index]))
                if len(i_to_p[index])>1:
                    label_items  = tokens_ids_tokenized[i_to_p[index][0]:(i_to_p[index][-1]+1)] 
                else:
                    label_items  = [tokens_ids_tokenized[i_to_p[index][-1]]]
                node_labels.append(label_items)
                
                if config_const.CONF_GRAPH_BUILDER_SELFLOOP in self.params and self.params[config_const.CONF_GRAPH_BUILDER_SELFLOOP]: 
                    self.insert_edge(index, index, "seq", edge_index, edge_feats)
                
                if index >0:
                    self.insert_edge(index-1, index, "seq", edge_index, edge_feats)
                    if config_const.CONF_GRAPH_BUILDER_BIDIRECTED in self.params and self.params[config_const.CONF_GRAPH_BUILDER_BIDIRECTED]:
                        self.insert_edge(index, index-1, "seq", edge_index, edge_feats)
                        
