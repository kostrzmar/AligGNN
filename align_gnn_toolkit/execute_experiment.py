import argparse
import warnings
import coloredlogs
from engine import EngineFactory
from torch.multiprocessing import  set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass
import os

warnings.filterwarnings("ignore") 
coloredlogs.install(fmt='[%(levelname)s] [%(asctime)s,%(msecs)03d] [(%(name)s[%(process)d)] [(%(threadName)s)] %(message)s', level='INFO')

EXPERIMENT_TEMPLATE = 'align_gnn_toolkit/experiments_repository/template_default.yaml'
EXPERIMENT_CONFIG = 'align_gnn_toolkit/experiments_repository/sick_experiments/sick_all_spda_test.yaml'

parser = argparse.ArgumentParser()
parser.add_argument('-temp', required=False, help='Path to the config file',nargs='?', const='1', type=str, default=EXPERIMENT_TEMPLATE)
parser.add_argument('-conf', required=False, help='Nbr of experiment from config',nargs='?', const='1', type=str, default=EXPERIMENT_CONFIG)
args = parser.parse_args()

if __name__ == '__main__':
    EngineFactory().startExecution(args)
