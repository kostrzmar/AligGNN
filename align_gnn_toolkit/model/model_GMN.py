import torch
from torch_geometric.nn import MessagePassing, BatchNorm, LayerNorm
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree
import torch.nn.functional as F
from torch_geometric.nn import MLP
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_scatter import scatter_mean, scatter_add

class GraphMatchingConvolution(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GraphMatchingConvolution, self).__init__(aggr=AttentionalAggregation(torch.nn.Linear(out_channels, 1), torch.nn.Linear(out_channels, out_channels)))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin_node = torch.nn.Linear(in_channels, out_channels)
        self.lin_message = torch.nn.Linear(out_channels * 2, out_channels)
        self.lin_passing = torch.nn.Linear(out_channels + in_channels, out_channels)
        self.batch_norm =LayerNorm(out_channels)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.lin_node.reset_parameters()
        self.lin_message.reset_parameters()
        self.lin_passing.reset_parameters()
        self.batch_norm.reset_parameters()
        

    def forward(self, x_s, edge_index_s, batch_s, x_t, edge_index_t, batch_t ):
        x_s_transformed = self.lin_node(x_s)
        x_t_transformed = self.lin_node(x_t)
        x=torch.cat([x_s_transformed, x_t_transformed], dim=-2)
        edge_index = torch.cat([edge_index_s, torch.add(edge_index_t, x_s.shape[0])], dim=-1)
        return self.propagate(edge_index, x=x, edge_index_s=edge_index_s,edge_index_t=edge_index_t, original_x_s=x_s, batch_s=batch_s, original_x_t=x_t, batch_t=batch_t)

    def message(self, edge_index_i, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=1)
        m = self.lin_message(x)
        return m

    def update(self, aggr_out, edge_index_s, edge_index_t, x, original_x_s, batch_s, original_x_t, batch_t):        
        attention_x_s, attention_x_t = self.compute_crosss_attention(original_x_s, batch_s, original_x_t, batch_t)
        attention_input_x_s = original_x_s - attention_x_s
        attention_input_x_t = original_x_t - attention_x_t  
        aggr_out_x_s = self.lin_passing(torch.cat([aggr_out[:batch_s.shape[0]], attention_input_x_s], dim=1))
        aggr_out_x_s = self.batch_norm(aggr_out_x_s)
        aggr_out_x_t = self.lin_passing(torch.cat([aggr_out[batch_s.shape[0]:], attention_input_x_t], dim=1))
        aggr_out_x_t = self.batch_norm(aggr_out_x_t)
        return aggr_out_x_s, edge_index_s, batch_s, aggr_out_x_t, edge_index_t, batch_t
    

    def compute_crosss_attention(self, x_s, batch_s, x_t, batch_t):
        original_x_s_b, mask_s = to_dense_batch(x_s, batch_s)
        original_x_t_b, mask_t = to_dense_batch(x_t, batch_t)
        B, MAX_N= mask_s.shape
        out_i = []
        out_j = []
        for index in range(B):
            x_i = original_x_s_b[index][mask_s[index]]
            x_j = original_x_t_b[index][mask_t[index]]
            a = self.pairwise_cosine_similarity(x_i, x_j)
            a_i = F.softmax(a, dim=1)
            a_j = F.softmax(a, dim=0)
            att_i = torch.matmul(a_i, x_j)
            att_j = torch.matmul(a_j.T, x_i)
            out_i.append(att_i)
            out_j.append(att_j)
        return torch.concat(out_i, dim=0), torch.concat(out_j, dim=0)
    
    def pairwise_cosine_similarity(self, a, b):
        a_norm = torch.norm(a, dim=1).unsqueeze(-1)
        b_norm = torch.norm(b, dim=1).unsqueeze(-1)
        return torch.matmul(a_norm, b_norm.T)
    
    def __repr__(self) -> str:
        channels_repr = ''
        if hasattr(self, 'in_channels') and hasattr(self, 'out_channels'):
            channels_repr = f'{self.in_channels}, {self.out_channels}'
        elif hasattr(self, 'channels'):
            channels_repr = f'{self.channels}'
        return f'{self.__class__.__name__}({channels_repr})'
