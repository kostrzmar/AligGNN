from graph_builder.graph_builder_type_abstract import GraphBuilderTypeAbstract
from data_set.data_set_processor import DataSetProcessor
class GraphBuilderSequenceMaster(GraphBuilderTypeAbstract):
    def __init__(self, 
                params=None,
                data_set=None,
                master_builder=None
                ) -> None:
        super(GraphBuilderSequenceMaster, self).__init__(params, "Master")


    def isNodeOHE(self):
        return True
        
    def isEdgeOHE(self):
        return True
    
    def getEdgeLabels(self):
        return {"master":"master"}

    def getEdgeLabelsToIndex(self):
        return {"master":0}

    def getNodeFeatsNbr(self):
        return len(self.getEdgeLabels())

    def getEdgeFeatsNbr(self): 
        return len(self.getEdgeLabels())


    def customize_graph(self, item, item_type, sent_node_feats, sent_edge_index, sent_node_labels,sent_edge_feats, graph_dict):
        unk_item = self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_def_unk_word()
        info_for_graphs = self.getGraphData()
        for info_for_graph in info_for_graphs:
            tokens_for_nodes = info_for_graph["tokens"]
            pos_one_hot = self.getOneHotEncoding(
                            self.getEdgeLabels(), 
                            self.getEdgeLabelsToIndex(), 
                            "master") 
            sent_node_feats.append(
                            self.update_node_feats(
                                self.init_node_feats(),
                                pos_one_hot))
            graph_dict["master"] = len(sent_node_feats)-1
            
            sent_node_labels.append([unk_item])
            
            index = len(sent_node_labels)-1
            for i in range(len(tokens_for_nodes)):
                sent_edge_index += [[i, index]]        
                sent_edge_feats.append(
                self.update_edge_feats(
                        self.init_edge_feats(),
                        self.getOneHotEncoding(
                            self.getEdgeLabels(), 
                            self.getEdgeLabelsToIndex(),"master")))

