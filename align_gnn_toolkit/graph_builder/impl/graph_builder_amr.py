from graph_builder.graph_builder_type_abstract import GraphBuilderTypeAbstract
from data_set.data_set_abstract import DatasetAbstract
from collections import defaultdict
from utils import config_const
from data_set.data_set_processor import DataSetProcessor
import torch
class GraphBuilderAMR(GraphBuilderTypeAbstract):
    def __init__(self, 
                params=None,
                data_set=None,
                master_builder=None
                ) -> None:
        super(GraphBuilderAMR, self).__init__(params, "Amr")
        self.build_rel_from_lm = self.get_parameter_as_bool(config_const.CONF_GRAPH_BUILDER_RELATION_FROM_LM)
        self.only_amr = self.get_parameter_as_bool(config_const.CONF_GRAPH_ONLY_ARM)
        self.is_multi_graph = self.get_parameter_as_bool(config_const.CONF_GRAPH_BUILDER_MULTIGRAPH)
        self.convert_relation_to_node = self.get_parameter_as_bool(config_const.CONF_GRAPH_BUILDER_RELATION_TO_NODE)

    def isNodeOHE(self):
        return True
        
    def isEdgeOHE(self):
        return True

    def getEdgeFeatsNbr(self): 
        emb_size= 0
        if self.build_rel_from_lm:
           emb_size = self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_vocab_embedding_size() 
           self.org_offset_edge_feats = self.offset_edge_feats
           self.offset_edge_feats = self.offset_edge_feats + emb_size 
        return emb_size + len(self.getEdgeLabels())
    

    def getNodeFeatsNbr(self):
        emb_size= 0
        if self.only_amr and len(self.params[config_const.CONF_GRAPH_BUILDER_NAME])==1:
            emb_size = self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_vocab_embedding_size()
            self.offset_node_feats = self.offset_node_feats + emb_size 
        return emb_size + len(self.getEdgeLabels())
    
    def add_bidirectional(self, to_id, from_id, edge_index, edge_feats):
        edge_id = self.get_edge_index([to_id, from_id], edge_index) 
        if edge_id <0:
            loc_edge_features = self.init_edge_feats() 
            loc_edge_features[0]=1 # add seq 
            edge_index += [[to_id,from_id]]
            edge_feats.append(loc_edge_features)
    
    def insert_edge(self, from_id, to_id, edge_type, edge_frame, edge_index, edge_feats, sent_node_feats,sent_node_labels, amr_node_features, i_to_arm_i, amr_key_to_i):
        edge_id  = -1
        new_node_index = None
        if self.is_multi_graph:
            edge_id = self.get_edge_index([from_id, to_id], edge_index)
            if edge_id>=0:
                edge_feats = edge_feats[edge_id]
                self.update_edge_feats(
                    edge_feats,
                        self.getOneHotEncoding(
                            self.getEdgeLabels(), 
                            self.getEdgeLabelsToIndex(), 
                            edge_type))
                
                
        if edge_id==-1:
            
            edge_features = self.init_edge_feats() 
            if not self.convert_relation_to_node and self.build_rel_from_lm:
                if edge_frame:
                    str_tokens = self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_def_tokenizer([edge_frame], is_per_words=True)
                    str_tokens_ids = [self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab[x] for x in str_tokens]        
                    self.initialize_feature_with_embedding(edge_features, self.org_offset_edge_feats, str_tokens_ids, edge_frame)
    
            if self.convert_relation_to_node:
                node_content = {}
                node_content['type']=":type"
                if edge_frame:
                    node_content['name'] = edge_frame
                else: 
                    if edge_type in self.getEdgeLabels():
                        node_content['name'] = self.getEdgeLabels()[edge_type]
                    else:
                        node_content['name'] = edge_type
                new_node_index = self.addNode( sent_node_feats,sent_node_labels, node_content)
                amr_key_to_i[str(from_id)+"_"+str(to_id)+"_"+edge_type] = new_node_index

                    
                    
                edge_index += [[from_id, new_node_index]]
                edge_feats.append(
                    self.update_edge_feats(
                        edge_features,
                            self.getOneHotEncoding(
                                self.getEdgeLabels(), 
                                self.getEdgeLabelsToIndex(), 
                                edge_type)))                           
                
                edge_index += [[new_node_index, to_id]]
                edge_feats.append(
                    self.update_edge_feats(
                        edge_features,
                            self.getOneHotEncoding(
                                self.getEdgeLabels(), 
                                self.getEdgeLabelsToIndex(), 
                                edge_type)))                            
                
            else:
                edge_index += [[from_id, to_id]]
                edge_feats.append(
                    self.update_edge_feats(
                        edge_features,
                            self.getOneHotEncoding(
                                self.getEdgeLabels(), 
                                self.getEdgeLabelsToIndex(), 
                                edge_type)))
            
            if config_const.CONF_GRAPH_BUILDER_BIDIRECTED in self.params and self.params[config_const.CONF_GRAPH_BUILDER_BIDIRECTED]:
                if not new_node_index:
                    self.add_bidirectional(to_id, from_id, edge_index, edge_feats)
                else:
                    self.add_bidirectional(to_id, new_node_index, edge_index, edge_feats)
                    self.add_bidirectional(new_node_index, from_id, edge_index, edge_feats)
                    
            
        return edge_id, new_node_index    
    
    def initialize_feature_with_embedding(self, features, offset, str_tokens_ids, content):
        out,indices, _ = self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_def_word_embedding(str_tokens_ids, [content], False)
        emb_avg = torch.mean(torch.stack(list(out.values())), dim=0)
        for index, item in enumerate(emb_avg): 
            features[offset+index] = item
    
    def addNode(self, sent_node_feats,sent_node_labels, node_content ):
        
        str_tokens = self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_def_tokenizer([node_content['name']], is_per_words=True)
        str_tokens_ids = [self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab[x] for x in str_tokens]
        
        node_features = self.init_node_feats()
        if len(str_tokens_ids)==0:
            unk_item = self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_def_unk_word()
            str_tokens_ids = [unk_item]
        else:
            initialize_with_embedding=True
            if initialize_with_embedding:
                self.initialize_feature_with_embedding(node_features, 0, str_tokens_ids, node_content['name'])
        
        pos_one_hot = self.getOneHotEncoding(
                        self.getEdgeLabels(), 
                        self.getEdgeLabelsToIndex(), 
                        ''.join(filter(lambda x: not x.isdigit(), node_content['type']))) 
        sent_node_feats.append(
                        self.update_node_feats(
                            node_features,
                            pos_one_hot))
        sent_node_labels.append(str_tokens_ids)
        
        new_node_index = len(sent_node_labels)-1        
        return new_node_index # index of new node



    def customize_graph(self, item, item_type, sent_node_feats, sent_edge_index, sent_node_labels, sent_edge_feats, graph_dict):
        if not self.only_amr:
            indices = item[self.data_set.get_token_name(item_type,DatasetAbstract.SENTENCE_TOKEN_VOCAB_ID_MASK)]
            i_to_p = defaultdict(list)
            for pos, index in enumerate(indices):
                i_to_p[index].append(pos)
            info_for_graphs =  self.getDataConverterProcessor().parseSentence(item[self.data_set.get_token_name(item_type,DatasetAbstract.SENTENCE_TOKEN_VOCAB_ID)], indices)
        else:
            info_for_graphs =  self.getDataConverterProcessor().parseSentence(None, None, sentence_as_string=item[self.data_set.get_token_name(item_type,DatasetAbstract.SENTENCE_TOKEN_NAME)])    

        amr_node_features = []
        i_to_arm_i = defaultdict(list)
        amr_key_to_i = defaultdict(list)


        for info_for_graph in info_for_graphs:
            matched_indexes = set()
            
            if self.only_amr:
                for nodes in info_for_graph["nodes"]:
                    if nodes["index"] or nodes["index"]==0:
                        matched_indexes.add(nodes["index"])
                sent_edge_index = []
                sent_edge_feats = []
                           

            for pos,node_content in enumerate(info_for_graph['nodes']):
                node_index_id = -1
                if  node_content["index"] or node_content["index"]==0:
                    position_id = [node_content["index"]]
                    amr_key_to_i[node_content["key"]] = position_id[0]
                    for id in position_id:
                        arm_one_hot = self.getOneHotEncoding(
                            self.getEdgeLabels(), 
                            self.getEdgeLabelsToIndex(),
                            ''.join(filter(lambda x: not x.isdigit(), node_content['type'])) 
                            )  
                        sent_node_feats[id] = self.update_node_feats(
                            sent_node_feats[id],
                            arm_one_hot)
                        node_index_id = id
                        
                else:
                    node_index_id = self.addNode( sent_node_feats,sent_node_labels, node_content)
                    amr_key_to_i[node_content["key"]] = node_index_id
                    matched_indexes.add(node_index_id)
                    
                if config_const.CONF_GRAPH_BUILDER_SELFLOOP in self.params and self.params[config_const.CONF_GRAPH_BUILDER_SELFLOOP]:                         
                    edge_id = self.get_edge_index([node_index_id, node_index_id], sent_edge_index)                     
                    if edge_id< 0:
                        loc_edge_features = self.init_edge_feats() 
                        loc_edge_features[0]=1 # add seq 
                        sent_edge_index += [[node_index_id,node_index_id]]
                        sent_edge_feats.append(loc_edge_features)                        
                    
            
            for edge_content in info_for_graph['edges']:

                src_ids = [amr_key_to_i[edge_content['src']]]
                trg_ids = [amr_key_to_i[edge_content['tgt']]]
                    
                    
                for src_id in src_ids:
                    for trg_id in trg_ids:
                        edge_id, new_node_index = self.insert_edge(src_id, trg_id, ''.join(filter(lambda x: not x.isdigit(), edge_content['edge_type'])), edge_content["frame"], sent_edge_index, sent_edge_feats, sent_node_feats,sent_node_labels, amr_node_features, i_to_arm_i, amr_key_to_i)
                        if new_node_index:
                             matched_indexes.add(new_node_index)
        if self.only_amr:
            new_sent_node_feats = []
            new_sent_node_id_to_old_id = {}
            new_sent_node_labels = []
            for index, node_features in enumerate(sent_node_feats):
                if index in matched_indexes:
                    new_sent_node_feats.append(node_features)
                    new_sent_node_id_to_old_id[index] = len(new_sent_node_feats)-1
                    new_sent_node_labels.append(sent_node_labels[index])
            new_sent_edge_index  = []
            new_sent_edge_feats = []     
            for index, edge_index in enumerate(sent_edge_index):
                
                if edge_index[0] in matched_indexes and edge_index[1] in matched_indexes:
                    new_sent_edge_index.append([new_sent_node_id_to_old_id[edge_index[0]], new_sent_node_id_to_old_id[edge_index[1]] ] )
                    new_sent_edge_feats.append(sent_edge_feats[index])  
            return (new_sent_node_feats, new_sent_edge_index, new_sent_node_labels, new_sent_edge_feats)                    
                 

                
          
    