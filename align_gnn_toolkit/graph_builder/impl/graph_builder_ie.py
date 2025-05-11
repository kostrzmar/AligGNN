from graph_builder.graph_builder_type_abstract import GraphBuilderTypeAbstract
from data_set.data_set_processor import DataSetProcessor

class GraphBuilderSequenceIE(GraphBuilderTypeAbstract):
    def __init__(self, 
                params=None,
                data_set=None,
                master_builder=None
                ) -> None:
        super(GraphBuilderSequenceIE, self).__init__(params, "IE")

    def isNodeOHE(self):
        return False
        
    def isEdgeOHE(self):
        return True        
    
    def getEdgeLabels(self):
        return {"subject":"subject", "object":"object", "relation":"relation"}

    def getEdgeLabelsToIndex(self):
        return {"subject":0, "object":1, "relation":2}

    def getNodeFeatsNbr(self):
        return len(self.getEdgeLabels())

    def getEdgeFeatsNbr(self): 
        return len(self.getEdgeLabels())
    
    
    def addNode(self, sentence_mask, sent_node_feats,sent_node_labels, type ):
        unk_item = self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_def_unk_word()
        
        pos_one_hot = self.getOneHotEncoding(
                        self.getEdgeLabels(), 
                        self.getEdgeLabelsToIndex(), 
                        type) 
        sent_node_feats.append(
                        self.update_node_feats(
                            self.init_node_feats(),
                            pos_one_hot))
          
        
        sent_node_labels.append(unk_item)
        return len(sent_node_labels)-1 # index of new node
    
    def linkToNode(self, ie,  sent_edge_feats, node_rel_index,sent_edge_index, sentence_mapping,ie_type, type):

        for x in range(ie[ie_type][0], ie[ie_type][1]):
            x_ids = [x]
            for x_id in x_ids:
                sent_edge_index += [x_id, node_rel_index]
                sent_edge_feats.append(
                    self.update_edge_feats(
                    self.init_edge_feats(),
                    self.getOneHotEncoding(
                        self.getEdgeLabels(), 
                        self.getEdgeLabelsToIndex(), 
                        type)))
    
    
    def customize_graph(self, item, item_type, sent_node_feats, sent_edge_index, sent_node_labels, sent_edge_feats, graph_dict):
        info_for_graphs = self.getGraphData()
        for info_for_graph in info_for_graphs:
        
            for ie in info_for_graph['ie_content']:

                node_rel_index = self.addNode( sent_node_feats,sent_node_labels, "relation")
                self.linkToNode(ie,  info_for_graph, sent_edge_feats, node_rel_index, sent_edge_index, "rel_span", "relation")
                node_obj_index = self.addNode( sent_node_feats,sent_node_labels, "object")
                self.linkToNode(ie,  info_for_graph, sent_edge_feats, node_obj_index, sent_edge_index, "obj_span", "object")
                node_sub_index = self.addNode( sent_node_feats,sent_node_labels, "subject")
                self.linkToNode(ie,  info_for_graph, sent_edge_feats, node_sub_index, sent_edge_index, "sub_span", "subject")
                sent_edge_index += [[node_rel_index, node_obj_index]]
                sent_edge_feats.append(
                    self.update_edge_feats(
                    self.init_edge_feats(),
                    self.getOneHotEncoding(
                        self.getEdgeLabels(), 
                        self.getEdgeLabelsToIndex(), 
                        "object")))
                sent_edge_index += [[node_rel_index, node_sub_index]]
                sent_edge_feats.append(
                    self.update_edge_feats(
                    self.init_edge_feats(),
                    self.getOneHotEncoding(
                        self.getEdgeLabels(), 
                        self.getEdgeLabelsToIndex(), 
                        "subject")))
                