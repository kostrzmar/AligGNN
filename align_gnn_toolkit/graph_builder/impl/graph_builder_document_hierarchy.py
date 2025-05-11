from graph_builder.graph_builder_type_abstract import GraphBuilderTypeAbstract
from data_set.data_set_processor import DataSetProcessor

class GraphBuilderDocumentHierarchy(GraphBuilderTypeAbstract):
    def __init__(self, 
                params=None,
                data_set=None,
                master_builder=None
                ) -> None:
        super(GraphBuilderDocumentHierarchy, self).__init__(params, "documentHierarchy")

    def isNodeOHE(self):
        return False
        
    def isEdgeOHE(self):
        return True
            
    def getEdgeLabels(self):
        return {"paragraph":"paragraph", "document":"document"}

    def getEdgeLabelsToIndex(self):
        return {"paragraph":0, "document":1}

    def getNodeFeatsNbr(self):
        return len(self.getEdgeLabels())

    def getEdgeFeatsNbr(self): 
        return len(self.getEdgeLabels())
    
    
    def addNode(self, sent_node_feats,sent_node_labels, type ):
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
    

    def linkToNode(self, x_ids, sent_edge_index, sent_edge_feats, type ):

        for x_id in x_ids:
            sent_edge_index += [[x_id[0], x_id[1]]]
            sent_edge_feats.append(
                self.update_edge_feats(
                self.init_edge_feats(),
                self.getOneHotEncoding(
                    self.getEdgeLabels(), 
                    self.getEdgeLabelsToIndex(), 
                    type)))
    
    
    def customize_graph(self, item, item_type, sent_node_feats, sent_edge_index, sent_node_labels, sent_edge_feats, graph_dict):
        pass
                