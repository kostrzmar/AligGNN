import argparse
import coloredlogs, logging
from engine import EngineFactory
from data_set import DataHolder
from utils import utils_graph
import warnings
import os
from pyvis.network import Network
from torch.multiprocessing import set_start_method
from data_set.data_set_processor import DataSetProcessor
import os
from utils import ConfigUtils, config_const
from data_set.data_set_factory import DataSetFactory
try:
     set_start_method('spawn')
except RuntimeError:
    pass

warnings.filterwarnings("ignore") 
coloredlogs.install(fmt='[%(levelname)s] [%(asctime)s,%(msecs)03d] [(%(name)s[%(process)d)] [(%(threadName)s)] %(message)s', level='INFO')

EXPERIMENT_TEMPLATE = 'align_gnn_toolkit/experiments_repository/template_default.yaml'
EXPERIMENT_CONFIG = 'align_gnn_toolkit/experiments_repository/example.yaml'

parser = argparse.ArgumentParser()
parser.add_argument('-temp', required=False, help='Path to the config file',nargs='?', const='1', type=str, default=EXPERIMENT_TEMPLATE)
parser.add_argument('-conf', required=False, help='Nbr of experiment from config',nargs='?', const='1', type=str, default=EXPERIMENT_CONFIG)
args = parser.parse_args()


def convertNodeAttributesToText(key, node_labels):
    text_line = "Node Id ["+str(key)+"]"
    for sub_key in node_labels[key]:
        text_line+=" ["+sub_key+" -> "+"".join(node_labels[key][sub_key])+"]"
    return text_line

def convertInOutEdgesToText(direction, node_edges, show_details):
    count=0
    text_line=""
    if direction in node_edges:
        for items in node_edges[direction]:
            count+=len(items['link'])
        text_line += f'\t{direction} [{count}]:\n'
        if show_details:
            for in_items in node_edges[direction]:
                text_line+="\t\t["+str(in_items)+"]\n"
    return text_line

def convertEdgeAttributesToText(key, node_connections, show_details=True):
    node_edges = node_connections[key]
    text_line=convertInOutEdgesToText('in', node_edges, show_details)
    text_line+=convertInOutEdgesToText('out', node_edges, show_details)
    return text_line


def showInfo(data_set, index):
    if index > len(data_set):
        index = len(data_set)-1
    graph = data_set[index]
    utils_graph.print_graph_info(graph.get_source())
    utils_graph.print_graph_info(graph.get_target())
    node_labels, edge_labels, node_connections =data_set.graph_builder.get_labels_ext(graph.get_source())
    
    graph.print_sentences()
    graph.print_sentences( data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab)
    for key in node_labels:
        print(convertNodeAttributesToText(key, node_labels))
        print(convertEdgeAttributesToText(key, node_connections, show_details=True))
                        

        #node_labels, edge_labels, node_connections =graph_builder.get_labels_ext(graph.get_source())

def displayGraph(data_set, index, directory_name):
    if index > len(data_set):
        index = len(data_set)-1
    graph = data_set[index]
    source_graph = graph.get_source()
    node_labels, edge_labels =data_set.graph_builder.get_labels(source_graph)
    node_labels_ex, edge_labels_ex, node_connections_ex =data_set.graph_builder.get_labels_ext(graph.get_source())
    

    #B983FF -> #94B3FD -> #94DAFF -> #99FEFF
    #46244C -> #712B75 -> #C74B50 -> #D49B54
    #52006A #CD113B #FF7600 #FFA900
    
    G = utils_graph.convert_to_mulit_metworkx(source_graph, to_undirected=False)
   # G = torch_geometric.utils.to_networkx(source_graph, to_undirected=False)
    
    for node in list(G.nodes):
        title = f'ID: {node}\n'
        text =""
        color = "#B983FF"
        for key in node_labels_ex[node]:
            if key == 'word':
                text+=f'{",".join(node_labels_ex[node][key])}\n'
                title+=f'Token Id: {int(source_graph.node_labels[node]) if source_graph.node_labels[node].shape==0 else [int(x) for x in source_graph.node_labels[node] if x>-1]}\n'
            elif key == 'POS':
                text+=f'({",".join(node_labels_ex[node][key])})'
                color = "#B983FF"
                #title+=f'POS Name: {utils.utils_processing.pos_tags[node_labels_ex[node][key][0]]}\n'
                title+=f'POS Name: {node_labels_ex[node][key][0]}\n'
            elif key == 'IE':
                text+=f'({",".join(node_labels_ex[node][key])})'
                color = "#94DAFF"
                title+=f'IE Name: {",".join(node_labels_ex[node][key])}\n'
                
            elif key == "Master":
                text+=f'({",".join(node_labels_ex[node][key])})'
                color = "#99FEFF"
                title+=f'Name: {",".join(node_labels_ex[node][key])}\n'    
                
            elif key == "AMR":
                text+=f'({",".join(node_labels_ex[node][key])})'
                color = "#C74B50"
                title+=f'Name: {",".join(node_labels_ex[node][key])}\n'                 

            elif key == 'Constituency':
                text+=f'({",".join(node_labels_ex[node][key])})'
                color = "#94B3FD"
                #title+=f'Constituency Name: {utils.utils_processing.constituency_tags[node_labels_ex[node][key][0]]}\n'
                title+=f'Constituency Name: {node_labels_ex[node][key][0]}\n'
        
        G.nodes[node]["label"] = text
        G.nodes[node]["shape"] = 'ellipse'
        G.nodes[node]["color"] = color
        G.nodes[node]["title"] = title
        
        # Edges
        
        for in_edge in list(G.in_edges(node)):
            iE_index = 0
            links = edge_labels_ex[in_edge]['link']
            for index, edge in enumerate(links):
                title = f'ID: {in_edge[0]}->{in_edge[1]}\n'
                text = ""
                prefix = ""
                
                text = edge_labels_ex[in_edge]['link'][index]  
                title+= f'Link: ({edge_labels_ex[in_edge]["link"][index]})\n'
                color = "#52006A"

                prefix= "L"
                if len(links) ==1 or (len(links)>1 and index==0):
                
                    if "Dependency" in edge_labels_ex[in_edge]:
                        prefix +="D"
                        color = "#CD113B"
                        #title+= f'Dependency: ({edge_labels_ex[in_edge]["Dependency"][0]}) {utils.utils_processing.dependency_tags[edge_labels_ex[in_edge]["Dependency"][0]]}\n'
                        title+= f'Dependency: ({edge_labels_ex[in_edge]["Dependency"][0]}) {edge_labels_ex[in_edge]["Dependency"][0]}\n'
                        
                        
                    if "IE" in edge_labels_ex[in_edge]:
                        prefix +="IE"
                        color = "#FF7600"
                        title+= f'IE: ({edge_labels_ex[in_edge]["IE"][iE_index]})\n'
                        iE_index+=1
                
                    if "Master" in edge_labels_ex[in_edge]:
                        prefix +="M"
                        color = "#FFA900"
                        title+= f'Master: ({edge_labels_ex[in_edge]["Master"][iE_index]})\n'
                        iE_index+=1
                        
                    if "AMR" in edge_labels_ex[in_edge]:
                        prefix +="A"
                        color = "#D49B54"
                        title+= f'AMR: ({edge_labels_ex[in_edge]["AMR"][iE_index]})\n'
                        iE_index+=1
                
                
                G[int(edge.split("->")[0])][int(edge.split("->")[1])][index]["label"] = f'{text}\n({prefix})'
                G[int(edge.split("->")[0])][int(edge.split("->")[1])][index]["title"] = title
                G[int(edge.split("->")[0])][int(edge.split("->")[1])][index]["color"] = color
            
                     
                    
            
    
    
    nt = Network(directed =True, 
             height="1200px", 
             #height="100%",
             width="100%", 
             bgcolor="#eeeeee", 
             select_menu=True, 
             filter_menu=True)
    nt.from_nx(G)
    #nt.set_edge_smooth('dynamic')
    nt.toggle_physics(True)
    ##nt.force_atlas_2based()
    #nt.show_buttons(filter_=['physics'])
    owd = os.getcwd()
    if not os.path.exists("./text_graph_visualisation"):
        os.makedirs("./text_graph_visualisation")
    os.chdir("./text_graph_visualisation/")
    path = f'{directory_name}_{index}.html'
    nt.show(path)
    os.chdir(owd)


        


if __name__ == '__main__':
    
    config_utils = EngineFactory().getConfigurationUtils(args)
    engine = EngineFactory().getEngineType(config_utils)
    parameters = engine.getProcessingParameters()      
    data_holder = DataSetFactory.get_data_holder(params=parameters) 
    
    showInfo(data_holder.train_data_set, 4)
    displayGraph(data_holder.train_data_set, 4, DataSetFactory.getDirectoryName(params=parameters))
    
    #showInfo(data_holder.test_data_set, 2)
    #displayGraph(data_holder.test_data_set, 2, data_holder.getDirectoryName(parameters))
    
    #showInfo(data_holder.validation_data_set, 1, data_holder.getDirectoryName(parameters))
    