from graph_builder.graph_builder_type_abstract import GraphBuilderTypeAbstract
from data_set.data_set_processor import DataSetProcessor


class GraphBuilderSequenceConstituency(GraphBuilderTypeAbstract):
    def __init__(self, 
                params=None,
                data_set=None,
                master_builder =None
                ) -> None:
        super(GraphBuilderSequenceConstituency, self).__init__(params, "Constituency")

    def isNodeOHE(self):
        return True
        
    def isEdgeOHE(self):
        return False

    def getNodeFeatsNbr(self):
        return len(self.getEdgeLabels())

    def getEdgeFeatsNbr(self): 
        return 0

    def customize_graph(self, item, item_type, sent_node_feats, sent_edge_index, sent_node_labels, sent_edge_feats, graph_dict):
        unk_item = self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_def_unk_word()
        info_for_graphs = self.getGraphData()
        for info_for_graph in info_for_graphs:
            s_nodes = sorted(info_for_graph['nodes'].items())
            
            for item in s_nodes:
                sent_node_feats.append(self.update_node_feats(
                                self.init_node_feats(),
                                self.getOneHotEncoding(self.getEdgeLabels(), 
                                self.getEdgeLabelsToIndex(), 
                                item[1])))
                
                sent_node_labels.append(unk_item)
                
            for edges in info_for_graph['edges']:
                source_id, target_id = None, None

                source_id = edges[0]
                target_id = edges[1]
                sent_edge_index += [[ source_id, target_id ]]        
                sent_edge_feats.append(self.update_edge_feats(
                                self.init_edge_feats(),
                                self.getOneHotEncoding({}, {}, "")))

    
    