from torch_geometric.data import Dataset
from datasets import concatenate_datasets
from data_set.data_set_processor import DataSetProcessor
import torch
import os
from tqdm.auto import tqdm
from data_set.pair_data import PairData
import logging   
import shutil       
from abc import abstractmethod 
from torch_geometric.profile import timeit
from utils import config_const
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool, Process
from threading import Thread
           
class DatasetAbstract(Dataset):
    
    SENTENCE_TOKEN_NAME             = "sentence_token_name"
    DOCUMENT_NAME                   = "document_name"
    DOCUMENT_LEVEL                  = "document_level"
    PARAGRAPH_ID                    = "paragraph_id"
    SENTENCE_TOKEN_KEY              = "sentence_token_key"
    SENTENCE_TOKEN_ALIGNMENT_KEY    = "sentence_token_alignment_key"
    PARAGRAPH_TOKEN_ALIGNMENT_KEY   = "paragraph_token_alignment_key"
    SCORE_TOKEN_NAME                = "score_token_name"
    SCORE_VALUE                     = "score_value"
    SOURCE                          = "source"
    TARGET                          = "target"
    TOKEN                           = "tokens"
    TOKEN_VOCAB_ID                  = "tokens_vocab_id"
    SENTENCE_TOKEN                  = "tokens"
    SENTENCE_TOKEN_VOCAB_ID         = "tokens_vocab_id"
    SENTENCE_TOKEN_VOCAB_ID_MASK    = "tokens_vocab_id_mask"
    DATA_SET_TYPE_TRAIN             = "data_set_type_train"
    DATA_SET_TYPE_VALIDATION        = "data_set_type_validation"
    DATA_SET_TYPE_TEST              = "data_set_type_test"

    
    def __init__(self, root, 
                 transform=None, 
                 pre_transform=None, 
                 pre_filter=None, 
                 data_set=None, 
                 type=None, params=None, 
                 graph_builder=None, 
                 data_set_processor=None):
        self.type = type
        self.root = root
        self.data_set = data_set
        self.params = params
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.data_set_type_name = None
        self.corpus_attribute_name =None
        self.corpus_score_labels = None
        self.corpus_score_normalization_const = None
        self.corpus_score_normalization_min = None
        self.corpus_score_normalization_max = None
        self.initialize_data_set_parameters()
        self.has_download_loc=False
        self.expectedFile = None
        self.graph_builder = graph_builder
        self.data_set_processor = data_set_processor
        if self.data_set_processor:
            self.data_set_processor.initializeProcessors(self) 
        if self.graph_builder:
            if self.get_from_parameters(config_const.CONF_MODEL_EMBED_ONE_HOT, default_value=False):
                self.graph_builder.initialize(self) 
            else:
                self.graph_builder.data_set = self 
        self.execute_parallel = False
        self.limit = None
        if self.type:
            self.process_folder_path = os.path.join(self.root, 'processed', self.type)
        else:
            self.process_folder_path = os.path.join(self.root, 'processed')
        self.execute_parallel = self.get_from_parameters(config_const.CONF_DATASET_PROCESS_PARALLEL)
        self.in_memory_only = self.get_from_parameters(config_const.CONF_DATASET_PROCESS_PARALLEL, default_value=False)
        if self.in_memory_only:
            self.data_set_storage = []
        self.limit = self.get_from_parameters(config_const.CONF_DATASET_LIMIT)
        if self.get_from_parameters(config_const.CONF_DATASET_REGENERATE_GRAPH, default_value=False):
            #if self.root and self.root.startswith("./data/"):
            if self.root:
                if os.path.exists(self.root):
                    logging.info(f'Remove previous version of graph data at {self.root}')
                    shutil.rmtree(self.root)
                    self.params[config_const.CONF_DATASET_REGENERATE_GRAPH] = False
                else: 
                    logging.info(f'Flag [data.holder_regenerate] set {self.params[config_const.CONF_DATASET_REGENERATE_GRAPH]} but folder not found at {self.root}') 
        
        super().__init__(root, transform, pre_transform, pre_filter)
    
    @abstractmethod
    def initialize_data_set_parameters(self):
        pass
    
    @property
    def processed_dir(self) -> str:
        return self.process_folder_path

    @property
    def raw_file_names(self):
        return ['vocab_obj.pth']
        
    @property
    def processed_file_names(self):
        if self.expectedFile:
            if self.limit:
                size = len(self.expectedFile)
                if size < self.limit:
                    self.limit = size-1
                return self.expectedFile[:self.limit]
            else:
                return self.expectedFile
        else:
            logging.info(f'Dataset:[{self.data_set}] type:[{self.type}] -> checking data consistency')
            expectedFile = []
            set = self.load_data_set()
            if set:
                size = len(set)
                if self.limit:
                    if size > self.limit:
                        size = self.limit
                for index in tqdm(range(size), desc=self.type):
                    expectedFile.append(f'data_{self.type}_{index}.pt')
                self.expectedFile = expectedFile
        return expectedFile
   
    def load_data_set(self):
        if self.type in [self.data_set_type_name[DatasetAbstract.DATA_SET_TYPE_TRAIN], 
                         self.data_set_type_name[DatasetAbstract.DATA_SET_TYPE_TEST],
                         self.data_set_type_name[DatasetAbstract.DATA_SET_TYPE_VALIDATION]]:
            
            return self.load_dataset_by_type(self.type)
        return None
       
    def get_from_parameters(self, property_name, default_value=None):
        if property_name in self.params:
            return self.params[property_name]
        return default_value


    def get_path_to_root_folder(self):
        return self.get_from_parameters("dataset.path_to_root")
        
    def get_text_processor(self):
        self.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR)
   
   
    def load_vocab(self):
        self.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab=torch.load(os.path.join(self.raw_dir,'vocab_obj.pth'))
        if os.path.exists(os.path.join(self.raw_dir,'vocab_emb_obj.pth')):
            self.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab_embedding = torch.load(os.path.join(self.raw_dir,'vocab_emb_obj.pth'))
            self.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab_embedding_size = self.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab_embedding[0].shape
        
    
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        try:
            data = torch.load(os.path.join(self.processed_dir, f'data_{self.type}_{idx}.pt'))
            if self.pre_transform is not None:
                data = self.pre_transform(data)        
        except RuntimeError as err:
            logging.error(f'Error loading data_{self.type}_{idx}.pt', err)
        #self.extend_graph(data)
        return data    
    
    
    def extend_graph(self,data):
        
        if data.x_s.shape[0] != data.x_t.shape[0]:
            if data.x_s.shape[0] > data.x_t.shape[0]:
                data.x_t = torch.cat((data.x_t, torch.zeros(data.x_s.shape[0]-data.x_t.shape[0], data.x_t.shape[1]).to(data.x_t.device)))
            else:
                data.x_s = torch.cat((data.x_s, torch.zeros(data.x_t.shape[0]-data.x_s.shape[0], data.x_s.shape[1]).to(data.x_s.device)))
        
    
    def get_graph(self, item, item_type):
        return self.graph_builder.get_graph(self, item, item_type)
    
    def generate_graph(self, data_set,   set_type):
        self.generatePairData(data_set,   set_type)
            
    def _process_data_set_by_type(self, set_type, data_set):
        with timeit(log=False) as t:
            data_set = self.enhance_token(data_set, self.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab.get_stoi())
        logging.info(f'Dataset:[{self.data_set}] type:[{self.type}] -> enhancement done in {self.convert_duration(type="second", duration=t.duration)}') 
        logging.info(f'Dataset:[{self.data_set}] type:[{self.type}] -> build graphs') 
        with timeit(log=False) as t:
            self.generate_graph(data_set,   set_type)
        logging.info(f'Dataset:[{self.data_set}] type:[{self.type}] -> graphs done in {self.convert_duration(type="second", duration=t.duration)}') 
            
    
    def process_data_set(self, set_type):
        if not self.has_download_loc:
            self.download()        
        logging.info(f'Dataset:[{self.data_set}] type:[{self.type}] -> build vectors') 
        if self.graph_builder:
            self.graph_builder.initialize(self) 
        if set_type == self.data_set_type_name[DatasetAbstract.DATA_SET_TYPE_TRAIN]:
            if not self.train_set:
                self.train_set  = self.get_data_set_by_type(DatasetAbstract.DATA_SET_TYPE_TRAIN)
            self._process_data_set_by_type(set_type, self.train_set)
        elif set_type == self.data_set_type_name[DatasetAbstract.DATA_SET_TYPE_VALIDATION]:
            if not  self.val_set:
                self.val_set  = self.get_data_set_by_type(DatasetAbstract.DATA_SET_TYPE_VALIDATION)
            self._process_data_set_by_type(set_type, self.val_set)
        elif set_type == self.data_set_type_name[DatasetAbstract.DATA_SET_TYPE_TEST]:
            if not self.test_set:
                self.test_set  = self.get_data_set_by_type(DatasetAbstract.DATA_SET_TYPE_TEST)
            self._process_data_set_by_type(set_type, self.test_set)        
        

    def get_train_data_set(self):
        return self.train_set 
    
    def get_validation_data_set(self):
        return self.val_set 
    
    def get_test_data_set(self):
        return self.test_set 
    
    @abstractmethod
    def load_dataset_by_type(self, type, limit=None):
        pass
    
    def _get_data_set_by_type(self, data_set_type, data_set):
        if not data_set:       
            if self.limit:
                data_set = self.load_dataset_by_type(f'{data_set_type}[:{self.limit}]', self.limit)
            else:
                data_set = self.load_dataset_by_type(data_set_type)
        return data_set
        
    def get_data_set_by_type(self, data_set_type):
        if data_set_type == DatasetAbstract.DATA_SET_TYPE_TRAIN and self.data_set_type_name[DatasetAbstract.DATA_SET_TYPE_TRAIN]:
            return self._get_data_set_by_type(self.data_set_type_name[data_set_type], self.train_set)
        elif data_set_type == DatasetAbstract.DATA_SET_TYPE_VALIDATION and self.data_set_type_name[DatasetAbstract.DATA_SET_TYPE_VALIDATION]:
            return self._get_data_set_by_type(self.data_set_type_name[data_set_type], self.val_set)
        elif data_set_type == DatasetAbstract.DATA_SET_TYPE_TEST and self.data_set_type_name[DatasetAbstract.DATA_SET_TYPE_TEST]:
            return self._get_data_set_by_type(self.data_set_type_name[data_set_type], self.test_set)
    
    def get_data_set_column_names(self):
        return [self.corpus_attribute_name[DatasetAbstract.SOURCE+DatasetAbstract.SENTENCE_TOKEN_NAME], 
                         self.corpus_attribute_name[DatasetAbstract.TARGET+DatasetAbstract.SENTENCE_TOKEN_NAME]]
        
    
    def make_vocab(self):
        text_processor =  self.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR)
        self.train_set = self.get_data_set_by_type(DatasetAbstract.DATA_SET_TYPE_TRAIN)
        self.val_set = self.get_data_set_by_type(DatasetAbstract.DATA_SET_TYPE_VALIDATION)
        self.test_set = self.get_data_set_by_type(DatasetAbstract.DATA_SET_TYPE_TEST)            
        logging.info(f'Build vocab [{self.data_set}]...')   
        all_datasets = []
        for item in [self.train_set, self.val_set, self.test_set]:
            if item:
                all_datasets.append(item)
        text_processor.build_vocables(concatenate_datasets(all_datasets), 
                                        self.get_data_set_column_names())
        logging.info(f'Vocab size: {len(text_processor.vocab)}')
        torch.save(text_processor.vocab, os.path.join(self.raw_dir,'vocab_obj.pth'))
        torch.save(text_processor.vocab_embedding, os.path.join(self.raw_dir,'vocab_emb_obj.pth'))
        
    
        
    def download(self):
        if not os.path.exists(os.path.join(self.raw_dir,'vocab_obj.pth')):
            logging.info(f'Download dataset [{self.data_set}]...')
            self.make_vocab()
        elif not os.path.exists(os.path.join(self.raw_dir,'vocab_emb_obj.pth')):
            logging.info(f'Build vocab embedding for [{self.data_set}]...')
            self.load_vocab()
            self.get_text_processor().build_vocab_embeddings(None)
            torch.save(self.get_text_processor().vocab_embedding, os.path.join(self.raw_dir,'vocab_emb_obj.pth'))
        else:
            logging.info(f'Load local vocab for [{self.data_set}]...')
            self.load_vocab()
        self.has_download_loc = True
    

    def enhance_token(self, data_set, stoi):
        def convert_to_token(row):
            if self.corpus_score_labels:
                row[DatasetAbstract.SCORE_VALUE] = self.corpus_score_labels[row[_score_token_name]]
            elif self.corpus_score_normalization_const is not None:
                row[DatasetAbstract.SCORE_VALUE] = float(row[_score_token_name] / self.corpus_score_normalization_const)
            elif self.corpus_score_normalization_min is not None and self.corpus_score_normalization_max is not None:
                row[DatasetAbstract.SCORE_VALUE] = float((row[_score_token_name]-self.corpus_score_normalization_min) / (self.corpus_score_normalization_max- self.corpus_score_normalization_min))
            else:
                row[DatasetAbstract.SCORE_VALUE] = row[_score_token_name]
            return row
        _score_token_name = self.get_token_name(None, DatasetAbstract.SCORE_TOKEN_NAME)
        return data_set.map(convert_to_token)   
    
    def get_token_name(self, type, name):
        if name in [DatasetAbstract.SENTENCE_TOKEN_NAME, DatasetAbstract.SCORE_TOKEN_NAME]:
            if type:
                return self.corpus_attribute_name[type+name]
            return self.corpus_attribute_name[name]
        if type:
            return type+name
        return name
    
    
    def normalize_features(self, node_features):
        if node_features.shape[0] >1:
            node_features_sum = np.array(node_features.cpu().sum(-1)) #  (N, FIN) -> (N,1)
            node_features_inv_sum = np.power(node_features_sum, -1).squeeze() # (N,1) -> (N) 
            node_features_inv_sum[np.isinf(node_features_inv_sum)] = 1.
            diagonal_inv_features_sum_matrix = sp.diags(node_features_inv_sum)
            normalize_node_features = diagonal_inv_features_sum_matrix.dot(node_features.cpu())
            return torch.from_numpy(normalize_node_features)   
        else:
            return node_features
    
    def _generatePairDataFromGraph(self, source_data, target_data, label):
        if self.get_from_parameters(config_const.CONF_GRAPH_BUILDER_NORMALIZE_FEATURES, default_value=False):
            source_data.x = self.normalize_features(source_data.x)
            source_data.edge_attr = self.normalize_features(source_data.edge_attr)
            target_data.x = self.normalize_features(target_data.x)
            target_data.edge_attr = self.normalize_features(target_data.edge_attr)
        
        if hasattr(source_data, "graph_dict") and "sent_embeddings" in source_data.graph_dict:
            pair_data = PairData(x_s=source_data.x, 
                            edge_index_s=source_data.edge_index, 
                            edge_attr_s=source_data.edge_attr,
                            node_labels_s=source_data.node_labels,
                            sent_embedding_s = source_data.graph_dict["sent_embeddings"],
                            x_t=target_data.x, 
                            edge_index_t=target_data.edge_index, 
                            edge_attr_t=target_data.edge_attr,
                            node_labels_t=target_data.node_labels,
                            sent_embedding_t = target_data.graph_dict["sent_embeddings"],
                            y=torch.tensor((label), dtype=torch.float))
        else: 
            pair_data = PairData(x_s=source_data.x, 
                            edge_index_s=source_data.edge_index, 
                            edge_attr_s=source_data.edge_attr,
                            node_labels_s=source_data.node_labels,
                            x_t=target_data.x, 
                            edge_index_t=target_data.edge_index, 
                            edge_attr_t=target_data.edge_attr,
                            node_labels_t=target_data.node_labels,
                            y=torch.tensor((label), dtype=torch.float))
        
        return pair_data
    
    def _generatePairData(self, item):
        source_data = self.get_graph(item, DatasetAbstract.SOURCE)
        target_data = self.get_graph(item, DatasetAbstract.TARGET)
        label = item[DatasetAbstract.SCORE_VALUE]
        return self._generatePairDataFromGraph(source_data, target_data, label)
        
    def _updateDataLocal(self, pair_data, type,idx):
        if os.path.exists(os.path.join(self.processed_dir, f'data_{type}_{idx}.pt')):
            torch.save(pair_data, os.path.join(self.processed_dir, f'data_{type}_{idx}.pt'))
        
    
    def _generateDataLocal(self, item, type,idx,total):
        if not os.path.exists(os.path.join(self.processed_dir, f'data_{type}_{idx}.pt')):
            pair_data =self._generatePairData(item)
            torch.save(pair_data, os.path.join(self.processed_dir, f'data_{type}_{idx}.pt'))
        print(f'{type}: {idx}/{total} [{idx/total*100:.2f}%]', end='\r', flush=True)
    
    def _generateDataLocalInRange(self, set, type,idx_from, idx_to, total):
        for idx in range(idx_from, idx_to):
            self._generateDataLocal(set[idx],type, idx, total)
    
    
    def generatePairData(self, set, type):        
        idx = 0
        total = len(set)
        if config_const.CONF_DATASET_NBR_PROCESSES in self.params and int(self.params[config_const.CONF_DATASET_NBR_PROCESSES])>1:
            nbr_of_processes = int(self.params[config_const.CONF_DATASET_NBR_PROCESSES])
            step_size = int(len(set)/nbr_of_processes)
            start_index = 0
            processes = []
            for index in range(nbr_of_processes):
                idx_from = start_index
                idx_to = start_index+step_size
                p = Process(target=self._generateDataLocalInRange, args=(set, type,idx_from, idx_to, total))
                processes.append(p)       
                start_index += step_size
            for process in processes:
                process.start()

            for process in processes:
                process.join()

        else:
            if config_const.CONF_DATASET_ID_FROM in self.params and self.params[config_const.CONF_DATASET_ID_FROM] and int(self.params[config_const.CONF_DATASET_ID_FROM])>1 and config_const.CONF_DATASET_ID_TO in self.params and self.params[config_const.CONF_DATASET_ID_TO] and int(self.params[config_const.CONF_DATASET_ID_TO])>int(self.params[config_const.CONF_DATASET_ID_FROM]): 
                self._generateDataLocalInRange(set, type,int(self.params[config_const.CONF_DATASET_ID_FROM]), int(self.params[config_const.CONF_DATASET_ID_TO]), total)
            else:
                for idx, item in enumerate(set):
                    self._generateDataLocal(item,type, idx,total)      
            

    def process(self):
        self.process_data_set(self.type)
        
    def print_info(self):
        count=0
        if self.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab_embedding:
            for voc in self.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab.get_itos():
                if self.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab_embedding[self.get_text_processor().vocab[voc]].sum().item() ==0:
                    count+=1
            print(f'OOE % :{count/len(self.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab):.3g}')
        else:
            print(f'Contextual embedding...')
        
    
    def convert_duration(self,type, duration):
        if type=="second":    
            minutes, seconds = divmod(duration, 60)
            hours, minutes = divmod(minutes, 60)
            return "%d:%02d:%02d" % (hours, minutes, seconds)
        return "Not supported ->" + type 