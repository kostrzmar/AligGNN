from data_set.data_set_abstract import DatasetAbstract
from data_set.text_data import TextData

           
class DatasetDocumentAbstract(DatasetAbstract):
    

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, data_set=None, type=None, params=None, graph_builder=None, data_set_processor=None):

        super().__init__(root, transform, pre_transform, pre_filter, data_set=data_set, type=type, params=params, graph_builder=graph_builder, data_set_processor=data_set_processor)
        
    def enhance_token(self, data_set, stoi):
        return data_set
    
    def generate_graph(self, data_set,   set_type):
        self.generateData(data_set,   set_type)
    

    def _generateData(self, item):
        sentence_data = self.get_graph(item, None)

        data = TextData(x=sentence_data.x, 
                            edge_index=sentence_data.edge_index, 
                            edge_attr=sentence_data.edge_attr,
                            node_labels=sentence_data.node_labels, 
                            y=sentence_data.y)
        return data
    

    