import argparse
import warnings
import coloredlogs, logging
from engine import EngineFactory
from torch.multiprocessing import set_start_method
from data_set.data_set_factory import DataSetFactory
from torch_geometric.loader import DataLoader
from utils import config_const

try:
     set_start_method('spawn')
except RuntimeError:
    pass

warnings.filterwarnings("ignore") 
coloredlogs.install(fmt='[%(levelname)s] [%(asctime)s,%(msecs)03d] [(%(name)s[%(process)d)] [(%(threadName)s)] %(message)s', level='INFO')

EXPERIMENT_TEMPLATE = 'align_gnn_toolkit/experiments_repository/template_default.yaml'
EXPERIMENT_CONFIG = 'align_gnn_toolkit/experiments_repository/pregenerate_de_dataset_train.yaml'

parser = argparse.ArgumentParser()
parser.add_argument('-temp', required=False, help='Path to the config file',nargs='?', const='1', type=str, default=EXPERIMENT_TEMPLATE)
parser.add_argument('-conf', required=False, help='Nbr of experiment from config',nargs='?', const='1', type=str, default=EXPERIMENT_CONFIG)
args = parser.parse_args()



if __name__ == '__main__':
    
    config_utils = EngineFactory().getConfigurationUtils(args)
    engine = EngineFactory().getEngineType(config_utils)

    config_utils = engine.getProcessingParameters()
    
    alignment_datasets = config_utils[config_const.CONF_ALIGNMENT_GRAPH_DATA_SET]
    config_utils[config_const.CONF_DATASET_NAME] = alignment_datasets[0]
    config_utils[config_const.CONF_DATASET_INIT_SINGLE_DATASET] ="test"
      
    data_holder = DataSetFactory.get_data_holder(params=config_utils) 
    
    alignment_test_loader =  data_holder.test_data_loader 
    
    alignment_test_loader = DataLoader(alignment_test_loader.dataset, 
            batch_size=1, 
            shuffle=False,
            follow_batch=alignment_test_loader.follow_batch, 
            exclude_keys=alignment_test_loader.exclude_keys
            )   
    
    for index, data in enumerate(alignment_test_loader):
       
        edge_index_s, x_s, edge_attr_s, x_s_batch =    data.edge_index_s, data.x_s, data.edge_attr_s, data.x_s_batch
        if edge_index_s[0].min()<0:
            print(index, "0")
            print(edge_index_s)
        if edge_index_s[1].min()<0:
            print(index, "1")
            print(edge_index_s)
            
    
    logging.info("Generation done")
    
    
    
        
                    

                
    
    