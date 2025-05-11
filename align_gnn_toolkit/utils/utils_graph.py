import networkx as nx
import torch
import torch_geometric
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Any,  Optional, Iterable, Union


def show_graph_extended(graph_data, 
                        node_labels=None, 
                        edge_labels=None, 
                        title=None, 
                        to_undirected=False):
    G = torch_geometric.utils.to_networkx(graph_data, to_undirected=to_undirected)

    plt.figure(3, figsize=(20,20))
    if title:
        plt.title(title)    
    pos=nx.spring_layout(G,seed=5)
    ax = plt.gca()
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="skyblue",node_size=1500,alpha=0.5)
    nx.draw_networkx_labels(G, pos, ax=ax, labels=node_labels, font_color="red", font_size=11)
    curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
    straight_edges = list(set(G.edges()) - set(curved_edges))
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=straight_edges, alpha=0.6)
    arc_rad = 0.15
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}', alpha=0.6)
    if edge_labels:
        edge_weights = edge_labels    
    else:
        edge_weights = nx.get_edge_attributes(G,'w')
    curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
    straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=curved_edge_labels,rotate=False, font_color='purple')
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=straight_edge_labels,rotate=False,  font_color='purple')
    plt.show()    





def show_graph(graph_data, to_undirected=False):
    g = torch_geometric.utils.to_networkx(graph_data, to_undirected=to_undirected)
    if hasattr(graph_data, 'y'):
        node_labels = graph_data.y[list(g.nodes)].numpy()
    
    plt.figure(1,figsize=(14,12)) 
    nx.draw(g, 
            cmap=plt.get_cmap('Set1'),
            node_color = node_labels,
            node_size=75,
            linewidths=6)
    plt.show()
    

def visualize(h, color=None, epoch=None, loss=None):
    plt.figure(figsize=(14,12))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        if isinstance(h, torch_geometric.data.Data):
            h = torch_geometric.utils.to_networkx(h, to_undirected=False)

        nx.draw_networkx(
            h, pos=nx.spring_layout(h, seed=42), with_labels=True,
            node_color=color, cmap="Set2"
        )
    plt.show()
    

    
    
def visualize_TSNE(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

def get_sentence(graph, vocab):
    if graph.node_labels.shape[0] ==1:
        return " ".join([ vocab.get_itos()[x] for x in graph.node_labels])
    else:
        sentence=""
        for node in graph.node_labels:
            out = ""
            for index in range(node.shape[0]):
                if node[index] != -1:
                    out+=vocab.get_itos()[node[index]]
            sentence+=out+" "
            
        return sentence
        

        self.addLabel(node_labels, index,out) 
    

    

def print_dataset_info(dataset):
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    if hasattr(dataset, 'num_classes'):
        print(f'Number of classes: {dataset.num_classes}')
    if hasattr(dataset, 'num_features'):  
        print(f'Number of features: {dataset.num_features}')
    if hasattr(dataset, 'num_node_features'):  
        print(f'Number of nodes features: {dataset.num_node_features}')
    if hasattr(dataset, 'num_edge_features'):  
        print(f'Number of edges features: {dataset.num_edge_features}')

    
    
def print_graph_info(data):
    # Gather some statistics about the graph.

    print(data)
    print(f'Number of graph s keys: {data.keys()}')
    print('======================')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    if "train_mask" in data.keys():
        print(f'Number of training nodes: {data.train_mask.sum()}')
        print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
    print(f'Contains self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    print(f'Edge_index shape: {data.edge_index.shape}')
    print(f'X shape: {data.x.shape}')
    if data.y:
        print(f'Y shape: {data.y.shape}')
    
    
def convert_to_mulit_metworkx(
    data: 'torch_geometric.data.Data',
    node_attrs: Optional[Iterable[str]] = None,
    edge_attrs: Optional[Iterable[str]] = None,
    graph_attrs: Optional[Iterable[str]] = None,
    to_undirected: Optional[Union[bool, str]] = False,
    remove_self_loops: bool = False,
    ) -> Any:
    G = nx.MultiGraph() if to_undirected else nx.MultiDiGraph()

    G.add_nodes_from(range(data.num_nodes))

    node_attrs = node_attrs or []
    edge_attrs = edge_attrs or []
    graph_attrs = graph_attrs or []

    values = {}
    for key, value in data(*(node_attrs + edge_attrs + graph_attrs)):
        if torch.is_tensor(value):
            value = value if value.dim() <= 1 else value.squeeze(-1)
            values[key] = value.tolist()
        else:
            values[key] = value

    to_undirected = "upper" if to_undirected is True else to_undirected
    to_undirected_upper = True if to_undirected == "upper" else False
    to_undirected_lower = True if to_undirected == "lower" else False

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):

        if to_undirected_upper and u > v:
            continue
        elif to_undirected_lower and u < v:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)

        for key in edge_attrs:
            G[u][v][key] = values[key][i]

    for key in node_attrs:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    for key in graph_attrs:
        G.graph[key] = values[key]

    return G