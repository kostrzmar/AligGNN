from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from model.convolutions import LEEConv
from torch_geometric.nn.pool.select import SelectTopK
from torch_geometric.nn.pool.connect import FilterEdges
from torch_geometric.utils import (
    add_remaining_self_loops,
    remove_self_loops,
    scatter,
    softmax,
    to_edge_index,
    to_torch_coo_tensor,
    to_torch_csr_tensor,
)

class ASAEPooling(torch.nn.Module):
    r"""The Adaptive Structure Aware Pooling operator from the
    `"ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical
    Graph Representations" <https://arxiv.org/abs/1911.07979>`_ paper.
    
    Extension about edge attributes. 

    Args:
        in_channels (int): Size of each input sample.
        ratio (float or int): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`, or the value
            of :math:`k` itself, depending on whether the type of :obj:`ratio`
            is :obj:`float` or :obj:`int`. (default: :obj:`0.5`)
        GNN (torch.nn.Module, optional): A graph neural network layer for
            using intra-cluster properties.
            Especially helpful for graphs with higher degree of neighborhood
            (one of :class:`torch_geometric.nn.conv.GraphConv`,
            :class:`torch_geometric.nn.conv.GCNConv` or
            any GNN which supports the :obj:`edge_weight` parameter).
            (default: :obj:`None`)
        edge_dim (int): The number of edge features     
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        add_self_loops (bool, optional): If set to :obj:`True`, will add self
            loops to the new graph connectivity. (default: :obj:`False`)
        **kwargs (optional): Additional parameters for initializing the
            graph neural network layer.
    """
    def __init__(self, in_channels: int, ratio: Union[float, int] = 0.5,
                 GNN: Optional[Callable] = None, edge_dim: int = None, dropout: float = 0.0,  
                 negative_slope: float = 0.2, add_self_loops: bool = False,
                 **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.GNN = GNN
        self.add_self_loops = add_self_loops

        self.lin = Linear(in_channels, in_channels)
        if edge_dim is not None:
            self.att = Linear(2 * in_channels+edge_dim, 1)
        else:
            self.att = Linear(2 * in_channels, 1)
        self.gnn_score = LEEConv(self.in_channels, 1, edge_dim=edge_dim)
        if self.GNN is not None:
            self.gnn_intra_cluster = GNN(self.in_channels, self.in_channels,
                                         **kwargs)
        else:
            self.gnn_intra_cluster = None

        self.select = SelectTopK(1, ratio)
        self.connect = FilterEdges()
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin.reset_parameters()
        self.att.reset_parameters()
        self.gnn_score.reset_parameters()
        if self.gnn_intra_cluster is not None:
            self.gnn_intra_cluster.reset_parameters()
        self.select.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Tensor]:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The node feature matrix.
            edge_index (torch.Tensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights.
                (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example. (default: :obj:`None`)

        Return types:
            * **x** (*torch.Tensor*): The pooled node embeddings.
            * **edge_index** (*torch.Tensor*): The coarsened edge indices.
            * **edge_weight** (*torch.Tensor, optional*): The coarsened edge
              weights.
            * **batch** (*torch.Tensor*): The coarsened batch vector.
            * **index** (*torch.Tensor*): The top-:math:`k` node indices of
              nodes which are kept after pooling.
        """
        N = x.size(0)

        edge_index, edge_attr = add_remaining_self_loops(
            edge_index, edge_attr, fill_value=1., num_nodes=N)

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x

        x_pool = x
        if self.gnn_intra_cluster is not None:
            x_pool = self.gnn_intra_cluster(x=x, edge_index=edge_index,
                                            edge_weight=edge_weight)

        x_pool_j = x_pool[edge_index[0]]
        x_q = scatter(x_pool_j, edge_index[1], dim=0, reduce='max')
        x_q = self.lin(x_q)[edge_index[1]]

        # Compute attention; include edge attributes if available
        if edge_attr is not None:
            att_input = torch.cat([x_q, x_pool_j, edge_attr], dim=-1)
        else:
            att_input = torch.cat([x_q, x_pool_j], dim=-1)

        score = self.att(att_input).view(-1)
        score = F.leaky_relu(score, self.negative_slope)
        score = softmax(score, edge_index[1], num_nodes=N)

        # Sample attention coefficients stochastically.
        score = F.dropout(score, p=self.dropout, training=self.training)

        v_j = x[edge_index[0]] * score.view(-1, 1)
        x = scatter(v_j, edge_index[1], dim=0, reduce='sum')

        # Cluster selection.
        fitness = self.gnn_score(x, edge_index=edge_index, edge_attr=edge_attr, edge_weight=edge_weight).sigmoid().view(-1)
        
        select_out = self.select(fitness, batch)
        perm = select_out.node_index
        weight = select_out.weight
        assert weight is not None   
        x = x[perm] * fitness[perm].view(-1, 1)
        
        connect_out = self.connect(select_out, edge_index, edge_attr, batch)
        return x, connect_out.edge_index, connect_out.edge_attr, edge_weight , connect_out.batch, perm, score


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'ratio={self.ratio})')