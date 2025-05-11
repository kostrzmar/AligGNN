from graph_builder.graph_builder_type_abstract import GraphBuilderTypeAbstract

class GraphBuilderSequenceDependency(GraphBuilderTypeAbstract):
    def __init__(self, 
                params=None,
                data_set=None,
                master_builder =None
                ) -> None:
        super(GraphBuilderSequenceDependency, self).__init__(params, "Dependency")
    
    def isNodeOHE(self):
        return False
        
    def isEdgeOHE(self):
        return True
    
    def getNodeFeatsNbr(self):
        return 0

    def getEdgeFeatsNbr(self): 
        return len(self.getEdgeLabels())

    def customize_graph(self, item, item_type, sent_node_feats, sent_edge_index, sent_node_labels, sent_edge_feats, graph_dict):

        info_for_graphs = self.getGraphData()
        for info_for_graph in info_for_graphs:
 
            for dependency in info_for_graph['graph_content']:

                src_ids = [dependency['src']]
                trg_ids = [dependency['tgt']]
                for src_id in src_ids:
                    for trg_id in trg_ids:
                        self.insert_edge(src_id, trg_id, dependency['edge_type'], sent_edge_index, sent_edge_feats)