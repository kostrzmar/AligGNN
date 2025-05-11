from graph_builder.data_converter_processor import DataConverterProcessor
import numpy as np
from abc import abstractmethod
import statistics
from utils import config_const

class GraphBuilderTypeAbstract():
    def __init__(self, 
                params=None,
                builder_name = None,
                ) -> None:
        self.params=params
        self.builder_name = builder_name
        self.data_set=None
        self.data_converter_processor = None
        self.indexToLabel = None
        self.offset_edge_feats = 0
        self.offset_node_feats = 0
        self.edge_feats_number = 0
        self.node_feats_number = 0
        self.total_edge_feature_number = 0
        self.total_node_feature_number = 0

    @abstractmethod   
    def customize_graph(self, item, item_type, sent_node_feats,sent_edge_index,sent_node_labels,sent_edge_feats, graph_dict=None):
        pass

    @abstractmethod
    def getEdgeLabels(self):
        return self.getDataConverterProcessor().tag_to_name(self.builder_name)
    

    @abstractmethod
    def getEdgeLabelsToIndex(self):
        return self.getDataConverterProcessor().tag_to_index(self.builder_name)
    
    @abstractmethod
    def get_tag_definition(self):
        return []
    
    @abstractmethod
    def getNodeFeatsNbr(self):
        return 0 
    
    @abstractmethod
    def getEdgeFeatsNbr(self): 
        return 0

    @abstractmethod
    def isNodeOHE(self):
        return False
        
    @abstractmethod
    def isEdgeOHE(self):
        return False

    def initializeGraphBuilder(self, data_set, data_converter_processor):
        self.data_set = data_set
        self.data_converter_processor = data_converter_processor

    def get_builder_name(self):
        return self.builder_name

    def getDataConverterProcessor(self):
        return self.data_converter_processor.getProcessor(self.builder_name)  
    

    def init_node_feats(self):
        return np.zeros(self.total_node_feature_number, dtype=int).tolist()

    def init_edge_feats(self):
        return np.zeros(self.total_edge_feature_number, dtype=int).tolist()

    def get_edge_index(self, edge, sent_edge_index):
        for index, edge_index in enumerate(sent_edge_index):
            if edge == edge_index:
                return index 
        return -1
    
    def insert_edge(self, from_id, to_id, edge_type, edge_index, edge_feats):
        edge_id  = -1
        if config_const.CONF_GRAPH_BUILDER_MULTIGRAPH in self.params and not self.params[config_const.CONF_GRAPH_BUILDER_MULTIGRAPH]:
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
            edge_index += [[from_id, to_id]]
            edge_feats.append(
                self.update_edge_feats(
                    self.init_edge_feats(),
                        self.getOneHotEncoding(
                            self.getEdgeLabels(), 
                            self.getEdgeLabelsToIndex(), 
                            edge_type)))
        return edge_id
        
    
    def update_node_feats(self, node_features, values, is_overdrive_zero=False, update_with_avergage=False):
        for index, item in enumerate(values):
            if item!=0: 
                if update_with_avergage:
                    node_features[self.offset_node_feats+index] = statistics.mean([node_features[self.offset_node_feats+index], item])
                else:
                    node_features[self.offset_node_feats+index] = item
        return node_features

    def update_edge_feats(self, edge_features, values, is_overdrive_zero=False):
        for index, item in enumerate(values):
            if item!=0:
                edge_features[self.offset_edge_feats+index] = item
        return edge_features
    
    def getOneHotEncoding(self, labels,name_to_index, value):
        _one_hot = np.zeros(len(labels), dtype=int)
        if value != "":
            if value in name_to_index:
                _one_hot[name_to_index[value]] =1
            else:
                print(f'OneHotEncoding failed for label {value} for builder {self.get_builder_name()}')
        return _one_hot
                

    def getEdgeIndexToLabel(self):
        if not self.indexToLabel:
            self.indexToLabel = self.convertToIndex(self.getEdgeLabelsToIndex())
        return self.indexToLabel 


    def convertToIndex(self, dic):
        out = {}
        for key in dic.keys():
            out[dic[key]] = key
        return out
    
        
    def convert_sentence_mask_to_mapping(self, sentence_mask):   
        sentence_mapping ={index:item for index,item in enumerate(sentence_mask)}
        return sentence_mapping
        
    def get_max_subwords(self, sentence_mask): 
        max_subwords = 1
        for i in set(sentence_mask):
            count = sentence_mask.count(i)
            if max_subwords < count: max_subwords = count  
        return max_subwords          
  
    def getGraphData(self):
        return self.getDataConverterProcessor().getData(self.builder_name)     
                     
        

    def addLabel(self, labels, key, value):
        if key not in labels.keys():
            labels[key] = value
        else:
            labels[key] = labels[key] + "\n" +value 
            
    def add_label_ext(self, labels, type, key, value):
        if key not in labels.keys():
            labels[key] = {}

        if type not in labels[key]:
            labels[key][type] = [value]
        else:
            labels[key][type].append(value) 
    
    def get_labels_ext(self, graph):
        node_labels = {}
        edge_labels = {}
        node_connections = {} 
        if not self.data_set.vocab:
            self.data_set.load_vocab()
            
        for index, node in enumerate(graph.node_labels):
            out = ""
            for index_node in range(node.shape[0]):
                if node[index_node] != -1:
                    out+=self.data_set.vocab.get_itos()[node[index_node]]
            self.add_label_ext(node_labels, "word",index,out) 
        
        
        last_embeding_pos = self.getNodeFeatsNbr()
        for extension_builder in self.extension_builders: 
            if type(self) is not type(extension_builder):
                if extension_builder.getNodeFeatsNbr()>0:
                    for index, x_emb in enumerate(graph.x):
                        for index_2, item in enumerate(x_emb[last_embeding_pos:last_embeding_pos+extension_builder.getNodeFeatsNbr()]): 
                            if item ==1:
                                self.add_label_ext(node_labels, 
                                    extension_builder.get_builder_name(),
                                    index,                                  
                                    f'{extension_builder.getEdgeIndexToLabel()[index_2]}')                  
                    last_embeding_pos+=extension_builder.getNodeFeatsNbr()    
                    
        for index in range(graph.edge_index.shape[1]):       
            self.add_label_ext(edge_labels, 
                               "link",
                               (graph.edge_index[0][index].item(), graph.edge_index[1][index].item()),
                               f'{graph.edge_index[0][index]}->{graph.edge_index[1][index]}')    
            last_embeding_pos = self.getEdgeFeatsNbr()
            for extension_builder in self.extension_builders:
                if type(self) is not type(extension_builder):     
                    if extension_builder.getEdgeFeatsNbr()>0:
                        one_hot_encoding = graph.edge_attr
                        for index_2, item in enumerate(one_hot_encoding[index][last_embeding_pos:last_embeding_pos+extension_builder.getEdgeFeatsNbr()]): 
                            if item ==1:
                                self.add_label_ext(edge_labels, 
                                    extension_builder.get_builder_name(),
                                    (graph.edge_index[0][index].item(), graph.edge_index[1][index].item()),
                                    f'{extension_builder.getEdgeIndexToLabel()[index_2]}')    
                        last_embeding_pos+=extension_builder.getEdgeFeatsNbr()
        
        for key in edge_labels.keys():
            self.add_label_ext(node_connections,
                    "out",
                    key[0],
                    edge_labels[key])
            self.add_label_ext(node_connections,
                    "in",
                    key[1],
                    edge_labels[key]) 
                 
        
        return node_labels, edge_labels, node_connections     

         
    def get_labels(self, graph):
        node_labels = {}
        edge_labels = {} 
        if not self.data_set.vocab:
            self.data_set.load_vocab()
        
        for index, node in enumerate(graph.node_labels):
            if isinstance(node, list):
                if len(node)==1:
                    self.addLabel(node_labels, index,f'{self.data_set.vocab.get_itos()[node[0]]}') 
                else:
                    out = ""
                    for index_node in range(len(node)):
                        if node[index_node] != -1:
                            out+=self.data_set.vocab.get_itos()[node[index_node]]
                    self.addLabel(node_labels, index,out) 
            else:
                if node.shape[0] == 1:
                    self.addLabel(node_labels, index,f'{self.data_set.vocab.get_itos()[node]}') 
                else:
                    out = ""
                    for index_node in range(node.shape[0]):
                        if node[index_node] != -1:
                            out+=self.data_set.vocab.get_itos()[node[index_node]]
                    self.addLabel(node_labels, index,out) 
    
        last_embeding_pos = self.getNodeFeatsNbr()
        for extension_builder in self.extension_builders: 
            if type(self) is not type(extension_builder):
                if extension_builder.getNodeFeatsNbr()>0:
                    for index, x_emb in enumerate(graph.x):
                        for index_2, item in enumerate(x_emb[last_embeding_pos:last_embeding_pos+extension_builder.getNodeFeatsNbr()]): 
                            if item ==1:
                                self.addLabel(node_labels, 
                                    index,
                                    f'({extension_builder.getEdgeIndexToLabel()[index_2]})')                  
                    last_embeding_pos+=extension_builder.getNodeFeatsNbr()      
                    
        
        for index in range(graph.edge_index.shape[1]):       
            self.addLabel(edge_labels, (graph.edge_index[0][index].item(), graph.edge_index[1][index].item()),f'{graph.edge_index[0][index]}->{graph.edge_index[1][index]}')    
            last_embeding_pos = self.getEdgeFeatsNbr()
            for extension_builder in self.extension_builders:
                if type(self) is not type(extension_builder):     
                    if extension_builder.getEdgeFeatsNbr()>0:
                        one_hot_encoding = graph.edge_attr
                        for index_2, item in enumerate(one_hot_encoding[index][last_embeding_pos:last_embeding_pos+extension_builder.getEdgeFeatsNbr()]): 
                            if item ==1:
                                self.addLabel(edge_labels, 
                                    (graph.edge_index[0][index].item(), graph.edge_index[1][index].item()),
                                    f'{extension_builder.getEdgeIndexToLabel()[index_2]}')    
                        last_embeding_pos+=extension_builder.getEdgeFeatsNbr()
        
  
        
        return node_labels, edge_labels
    
    def get_parameter_as_bool(self, param_name, default_value=False):
        out = None
        if param_name in self.params:
            out = self.params[param_name]
            assert isinstance(out, bool), f'{out} need to be boolean'
        else:
            out = default_value
        return out
