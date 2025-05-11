
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from model import AbstractModel

class GCN(AbstractModel):
    def __init__(self, feature_size, edge_feature_size, model_params=None):
        super(GCN, self).__init__(feature_size, edge_feature_size, model_params)
        
        
    def initialize_model(self, feature_size, edge_feature_size, model_params=None):   
        self.conv1 = GCNConv(in_channels=feature_size, out_channels=self.embedding_size)
        self.conv2 = GCNConv(in_channels=self.embedding_size, out_channels=int(self.embedding_size/2))
        self.conv3 = GCNConv(in_channels=int(self.embedding_size/2), out_channels=int(self.embedding_size/2))
        
        self.linear1 = Linear(2*self.embedding_size, self.embedding_size)
        self.linear2 = Linear(2*int(self.embedding_size/2), self.embedding_size)  
        self.out = Linear(6*int(self.embedding_size/2), self.output_dim)    
    
    
    def do_convolution(self, conv, x, edge_index,edge_attr):
        return conv(x, edge_index)