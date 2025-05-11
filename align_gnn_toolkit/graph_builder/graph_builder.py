from graph_builder.data_converter_processor import DataConverterProcessor
from data_set.data_set_processor import DataSetProcessor
from torch_geometric.data import Data
import numpy as np
import torch
import logging
from utils import config_const
class GraphBuilder():
    def __init__(self, 
                params=None,
                ) -> None:
        self.params=params
        self.data_set = None
        self.builders = []
        self.builders_to_id = {}
        self.data_converter_processor = DataConverterProcessor(params=params)
        self.initalized = False  
    
    def registerBuilder(self, builder):
        self.builders.append(builder)
        self.builders_to_id[builder.builder_name] = len(self.builders)-1
          
    def getBuilderByName(self, builder_name):
        return  self.builders[self.builders_to_id[builder_name]]
        
    def getRegisteredBuildersName(self):
        return [builder.builder_name for builder in self.builders]
    
    def initialize(self, data_set):
        if not self.initalized:
            self.getDataConverterProcessor().initializeProcessors(data_set,  self.getRegisteredBuildersName())
            self.initializeBuilders(data_set)
            if not self.data_set:
                self.data_set = data_set
            self.initalized = True
       
    def getDataConverterProcessor(self):
        return self.data_converter_processor


    def initializeBuilders(self, data_set):
        logging.info(f'Initializing [{len(self.builders)}] graph builders for [{data_set.data_set} -> {data_set.type}]')
        total_edge_feature_number = 0
        total_node_feature_number = 0
        for builder in self.builders:
            builder.initializeGraphBuilder(data_set, self.getDataConverterProcessor())
            builder.edge_feats_number = builder.getEdgeFeatsNbr()
            builder.node_feats_number = builder.getNodeFeatsNbr()
            total_edge_feature_number += builder.edge_feats_number
            total_node_feature_number += builder.node_feats_number
        
        for index, builder in enumerate(self.builders):
            builder.total_edge_feature_number = total_edge_feature_number
            builder.total_node_feature_number = total_node_feature_number
            if index == 0:
                builder.offset_edge_feats =0
                builder.offset_node_feats =0   
            else:
                builder.offset_edge_feats = self.builders[index-1].offset_edge_feats + self.builders[index-1].getEdgeFeatsNbr()
                builder.offset_node_feats = self.builders[index-1].offset_node_feats + self.builders[index-1].getNodeFeatsNbr()
            logging.info(f'Registered graph builder: [{builder.builder_name}], features: edge [{builder.getEdgeFeatsNbr()}], node [{builder.getNodeFeatsNbr()}],  offset: edge [{builder.offset_edge_feats}], edge [{builder.offset_node_feats}]')
        logging.info(f'Total features: edge [{total_edge_feature_number}], node [{total_node_feature_number}]')
  


    def get_graph_data(self, sent_node_feats, sent_edge_index, sent_node_labels, sent_edge_feats, graph_dict):
        sent_node_feats = np.asarray(sent_node_feats)
        sent_node_feats = torch.tensor(sent_node_feats, dtype=torch.float)
        
        sent_edge_feats = np.asarray(sent_edge_feats)
        sent_edge_feats = torch.tensor(sent_edge_feats, dtype=torch.float)
        
        sent_edge_index = torch.tensor(sent_edge_index)
        sent_edge_index = sent_edge_index.t().to(torch.long).view(2, -1)
        
        max_length = max(map(len,sent_node_labels))
        for index, i in enumerate(sent_node_labels):
            sent_node_labels[index] = np.pad(i, (0, max_length-len(i)), 'constant', constant_values=(0, -1)) 
        sent_node_labels = torch.from_numpy(np.array(sent_node_labels))           
        if graph_dict and len(graph_dict.keys())>0:
            return Data(x=sent_node_feats,
                        edge_index=sent_edge_index,
                        node_labels=sent_node_labels,
                        edge_attr = sent_edge_feats,
                        graph_dict = graph_dict
                        )
        return Data(x=sent_node_feats,
                        edge_index=sent_edge_index,
                        node_labels=sent_node_labels,
                        edge_attr = sent_edge_feats)

    def get_graph(self, dataset, item, item_type):
        sent_node_feats = []
        sent_edge_feats = [] 
        sent_edge_index = []
        sent_node_labels = []
        graph_dict = {}
        self.getDataConverterProcessor().preprocess_graph(dataset, item, item_type)
        for builder in self.builders:
            updated = builder.customize_graph(item, item_type, sent_node_feats,sent_edge_index,sent_node_labels,sent_edge_feats, graph_dict)
            if updated:
                sent_node_feats = updated[0] 
                sent_edge_index = updated[1] 
                sent_node_labels = updated[2] 
                sent_edge_feats = updated[3]
        graph = self.get_graph_data(sent_node_feats, sent_edge_index, sent_node_labels,sent_edge_feats, graph_dict)
        return graph
    
    def initialize_text_processor(self):
        self.data_set.load_vocab()
        self.initialize(self.data_set)
        
    
    
    def getGraphOffsets(self):
        node_offsets = []
        edge_offsets = []
        for builder in self.builders:            
            node_offsets.append((builder.isNodeOHE(),  builder.offset_node_feats,builder.node_feats_number))
            edge_offsets.append((builder.isEdgeOHE(),  builder.offset_edge_feats,builder.edge_feats_number))
        return node_offsets, edge_offsets
       
    def _convert_itos(self, item, vocab=None, ignore_unk=False, remove_roberta=False, include_index=False):

        out = ""
        if len(item) ==1:
            if ignore_unk:
                if item[0] == vocab.get_default_index():
                    out= ""
            out= vocab.get_itos()[item[0]]
        else:
            out= "".join([vocab.get_itos()[x] if x!= vocab.get_default_index() and x!=-1 else "" for x in item ])
        
        if remove_roberta:
            out = out.replace('Ġ', '')
        return out
    
    def get_labels_ext(self, graph):
        node_labels = {}
        edge_labels = {}
        node_connections = {} 
        is_normalize = False
        if config_const.CONF_GRAPH_BUILDER_NORMALIZE_FEATURES in  self.params and  self.params[config_const.CONF_GRAPH_BUILDER_NORMALIZE_FEATURES]:
            is_normalize=True
                
        
        if not self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab:
            self.initialize_text_processor()
                        
        for index, node in enumerate(graph.node_labels):
            out = self._convert_itos(node, self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab, ignore_unk=False, remove_roberta=True)
            

            self.add_label_ext(node_labels, "word",index,out) 
        
        last_embeding_pos = 0
        for builder in self.builders:
            if builder.getNodeFeatsNbr()>0:
                for index, x_emb in enumerate(graph.x):
                    offset = 0
                    node2labels = builder.getEdgeIndexToLabel()
                    if builder.getNodeFeatsNbr() > len(node2labels):
                        offset =  builder.getNodeFeatsNbr() - len(node2labels)
                    one_hot_encoding = x_emb[last_embeding_pos+offset:last_embeding_pos+offset+len(node2labels)]
                    for index_2, item in enumerate(one_hot_encoding): 
                        if (not is_normalize and item ==1)  or  (is_normalize and item>0):
                            self.add_label_ext(node_labels, 
                                builder.get_builder_name(),
                                index,                                  
                                f'{builder.getEdgeIndexToLabel()[index_2]}')                  
                last_embeding_pos+=builder.getNodeFeatsNbr()    
                    
        for index in range(graph.edge_index.shape[1]):       
            self.add_label_ext(edge_labels, 
                               "link",
                               (graph.edge_index[0][index].item(), graph.edge_index[1][index].item()),
                               f'{graph.edge_index[0][index]}->{graph.edge_index[1][index]}')    
            last_embeding_pos = 0
            for builder in self.builders:
                if type(self) is not type(builder):     
                    if builder.getEdgeFeatsNbr()>0:
                        one_hot_encoding = graph.edge_attr
                        offset = 0
                        edge2lab = builder.getEdgeIndexToLabel()
                        if builder.getEdgeFeatsNbr() > len(edge2lab):
                            offset = builder.getEdgeFeatsNbr() - len(edge2lab)
                        for index_2, item in enumerate(one_hot_encoding[index][last_embeding_pos+offset:last_embeding_pos+offset+builder.getEdgeFeatsNbr()]): 
                            if (not is_normalize and item ==1)  or  (is_normalize and item>0):
                                self.add_label_ext(edge_labels, 
                                    builder.get_builder_name(),
                                    (graph.edge_index[0][index].item(), graph.edge_index[1][index].item()),
                                    f'{builder.getEdgeIndexToLabel()[index_2]}')    
                        last_embeding_pos+=builder.getEdgeFeatsNbr()
        
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
            
            
    def get_labels(self, graph):
        node_labels = {}
        edge_labels = {} 
        if not self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab:
            self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).load_vocab()
        
        for index, node in enumerate(graph.node_labels):
            if isinstance(node, list):
                if len(node) == 1:
                    self.addLabel(node_labels, index,f'{self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab.get_itos()[node[0]]}') 
                else:
                    out = ""
                    for index_node in range(len(node)):
                        if node[index_node] != -1:
                            out+=self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab.get_itos()[node[index_node]]
                    self.addLabel(node_labels, index,out) 
            
            else:
                if node.shape[0] == 1:
                    self.addLabel(node_labels, index,f'{self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab.get_itos()[node]}') 
                else:
                    out = ""
                    for index_node in range(node.shape[0]):
                        if node[index_node] != -1:
                            out+=self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab.get_itos()[node[index_node]]
                    self.addLabel(node_labels, index,out) 
    
        last_embeding_pos = 0
        for builder in self.builders: 
            if builder.getNodeFeatsNbr()>0:
                for index, x_emb in enumerate(graph.x):
                    for index_2, item in enumerate(x_emb[last_embeding_pos:last_embeding_pos+builder.getNodeFeatsNbr()]): 
                        if item ==1:
                            offset = 0
                            edge2lab = builder.getEdgeIndexToLabel()
                            if builder.getNodeFeatsNbr() > len(edge2lab):
                                offset = 0 - builder.getNodeFeatsNbr() + len(edge2lab)                            
                            self.addLabel(node_labels, 
                                index,
                                f'({builder.getEdgeIndexToLabel()[offset+index_2]})')                  
                last_embeding_pos+=builder.getNodeFeatsNbr()      
                    
        
        for index in range(graph.edge_index.shape[1]):       
            self.addLabel(edge_labels, (graph.edge_index[0][index].item(), graph.edge_index[1][index].item()),f'{graph.edge_index[0][index]}->{graph.edge_index[1][index]}')    
            last_embeding_pos = 0
            for builder in self.builders: 
                if builder.getEdgeFeatsNbr()>0:
                    one_hot_encoding = graph.edge_attr
                    for index_2, item in enumerate(one_hot_encoding[index][last_embeding_pos:last_embeding_pos+builder.getEdgeFeatsNbr()]): 
                        if item ==1:
                            offset = 0
                            edge2lab = builder.getEdgeIndexToLabel()
                            if builder.getEdgeFeatsNbr() > len(edge2lab):
                                offset = 0 - builder.getEdgeFeatsNbr() + len(edge2lab)
                            self.addLabel(edge_labels, 
                                (graph.edge_index[0][index].item(), graph.edge_index[1][index].item()),
                                f'{builder.getEdgeIndexToLabel()[offset+index_2]}')    
                    last_embeding_pos+=builder.getEdgeFeatsNbr()
        
  
        
        return node_labels, edge_labels
    
    
    def extract_graph_by_builder(self, builder_name):
        pass