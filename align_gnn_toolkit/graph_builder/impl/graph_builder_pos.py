from graph_builder.graph_builder_type_abstract import GraphBuilderTypeAbstract

class GraphBuilderSequencePOS(GraphBuilderTypeAbstract):
    def __init__(self, 
                params=None,
                data_set=None,
                master_builder=None
                ) -> None:
        super(GraphBuilderSequencePOS, self).__init__(params,  "POS")

    def isNodeOHE(self):
        return True
        
    def isEdgeOHE(self):
        return False
    
    def getNodeFeatsNbr(self):
        return len(self.getEdgeLabels())
    

    def customize_graph(self, item, item_type, sent_node_feats, sent_edge_index, sent_node_labels, sent_edge_feats, graph_dict):
        info_for_graphs = self.getGraphData()
        for info_for_graph in info_for_graphs:
            for node_content in info_for_graph['node_content']:
                position_id = [node_content["position_id"]]
                for id in position_id:

                    pos_one_hot = self.getOneHotEncoding(
                        self.getEdgeLabels(), 
                        self.getEdgeLabelsToIndex(), 
                        node_content['pos'])  

                    if id >= len(sent_node_feats):
                        print("uppss")
                    sent_node_feats[id] = self.update_node_feats(
                        sent_node_feats[id],
                        pos_one_hot)        

          
    