
from data_set.impl.data_set_sick import SickDataset
from data_set.impl.data_set_sick_binary import SickBinaryDataset
from data_set.impl.data_set_sick_binary_45_thr import SickBinary45ThrDataset
from data_set.impl.data_set_sts import StsDataset
from data_set.impl.data_set_sts12 import Sts12Dataset
from data_set.impl.data_set_sts13 import Sts13Dataset
from data_set.impl.data_set_sts14 import Sts14Dataset
from data_set.impl.data_set_sts15 import Sts15Dataset
from data_set.impl.data_set_sts16 import Sts16Dataset
from data_set.impl.data_set_sts_multi import StsMultiDataset
from data_set.impl.data_set_newsela import NewselaDataset
from data_set.impl.data_set_newsela_binary import NewselaBinaryDataset
from data_set.impl.data_set_newsela_document import NewselaDocumentDataset
from data_set.impl.data_set_newsela_test import NewselaTestDataset
from data_set.impl.data_set_test import TestDataset
from data_set.impl.data_set_capito_sample import CapitoSampleDataset
from data_set.data_set_abstract import DatasetAbstract
from graph_builder.graph_builder_factory import GraphBuilderFactory
from data_set.data_set_processor import DataSetProcessor
from data_set.impl.data_set_custom import CustomDataset
from data_set.impl.data_set_custom_document import CustomDocumentDataset
from data_set.impl.data_set_capito import CapitoDataset
from data_set.impl.data_set_capito_positive_only import CapitoPositiveDataset
from data_set.impl.data_set_nli_neutral_only import NLIDataset
from data_set.impl.data_set_xnli_neutral_only import XNLIDataset
from data_set.impl.data_set_wiki_test import WikiDataset
from data_set.impl.data_set_GermanEval2022 import GermanEval2022Dataset
from data_set.impl.data_set_GermanEval2022_test import GermanEval2022DatasetTest
from data_set.impl.data_set_benchmark_test import BenchmarkTestDataset
from data_set.impl.data_set_sick_hetero import SickHeteroDataset
from data_set.impl.lazar_scale_binarize_transform import Binarize
from utils import config_const
from data_set.data_holder import DataHolder
import os

class DataSetFactory:
    
    @staticmethod
    def get_dataset_by_type(data_set, types, root, params, pre_transform=None):
        train_dataset, test_dataset, validation_dataset = None, None, None 
        graph_builder = GraphBuilderFactory.getBuilder(params=params)
        data_set_processor = DataSetProcessor(params=params)
        for type in types:
            if type =='train':
                train_dataset = data_set(root=root, type=type, params=params, graph_builder=graph_builder, data_set_processor=data_set_processor, pre_transform=pre_transform)
            elif type=='test':
                test_dataset = data_set(root=root, type=type, params=params, graph_builder=graph_builder, data_set_processor=data_set_processor, pre_transform=pre_transform)
            elif type=='validation':
                validation_dataset = data_set(root=root, type=type, params=params, graph_builder=graph_builder, data_set_processor=data_set_processor, pre_transform=pre_transform)
        return train_dataset, test_dataset, validation_dataset
    
    @staticmethod
    def getDataSet(root, data_set, params) -> DatasetAbstract: 
        train_dataset, test_dataset, validation_dataset = None, None, None
        types = ["train", "test", "validation"]
        if config_const.CONF_DATASET_INIT_SINGLE_DATASET in params and params[config_const.CONF_DATASET_INIT_SINGLE_DATASET]:
            types = [params[config_const.CONF_DATASET_INIT_SINGLE_DATASET]]
   
        if data_set == "sick":
            pre_transform = None
            if config_const.CONF_DATASET_BINARIZE in params and params[config_const.CONF_DATASET_BINARIZE]:
                pre_transform=Binarize(SickDataset)
            train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(SickDataset, types, root, params, pre_transform=pre_transform)                

        elif data_set == "sick_binary":
            train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(SickBinaryDataset, types, root, params)        
            
        elif data_set == "sick_binary45th":
            train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(SickBinary45ThrDataset, types, root, params)                
                    
        elif data_set == "stsb":
            train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(StsDataset, types, root, params)     
        
        elif data_set == "sts12":
            train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(Sts12Dataset, types, root, params)     
        
        elif data_set == "sts13":
            train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(Sts13Dataset, types, root, params)   
    
        elif data_set == "sts14":
            train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(Sts14Dataset, types, root, params)   
    
        elif data_set == "sts15":
            train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(Sts15Dataset, types, root, params)   
            
        elif data_set == "sts16":
            train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(Sts16Dataset, types, root, params)               
            
        elif data_set == "stsmulti":
            train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(StsMultiDataset, types, root, params)  
            
            
        elif data_set == "newsela":
            train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(NewselaDataset, types, root, params)
 
        elif data_set == "newsela_binary":
            train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(NewselaBinaryDataset, types, root, params)
        
        elif data_set == "newsela_document":
            train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(NewselaDocumentDataset, types, root, params)

        elif data_set == "test":
            train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(TestDataset, types, root, params)     
        
        elif data_set == "custom":
            train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(CustomDataset, types, root, params)   
    
        elif data_set == "custom_document":
            train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(CustomDocumentDataset, types, root, params)       
    
        elif data_set == "capito_sample":
            train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(CapitoSampleDataset, types, root, params)
       
        elif data_set == "nli_neutral_only":
            train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(NLIDataset, types, root, params)
            
        elif data_set == "capito":
            train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(CapitoDataset, types, root, params)
        
        elif data_set == "capito_pos":
            train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(CapitoPositiveDataset, types, root, params)
        
        elif data_set == "xnli":
            train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(XNLIDataset, types, root, params)

        elif data_set == "wiki_test":
            train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(WikiDataset, types, root, params)        

        elif data_set == "newsela_test":
            train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(NewselaTestDataset, types, root, params)    

        elif data_set == "german_eval_2022_test":
           train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(GermanEval2022DatasetTest, types, root, params)                

        elif data_set == "german_eval_2022":
           train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(GermanEval2022Dataset, types, root, params)                

        elif data_set == "benchmark_test":
           train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(BenchmarkTestDataset, types, root, params)                
        
        elif data_set == "sick_hetero":
           train_dataset, test_dataset, validation_dataset= DataSetFactory.get_dataset_by_type(SickHeteroDataset, types, root, params)                

        return  train_dataset, test_dataset, validation_dataset       
    
    @staticmethod        
    def get_data_holder(params, root=None): 
        if root:
            params[config_const.CONF_DATASET_PATH_TO_ROOT] = root
        else:
            root = params[config_const.CONF_DATASET_PATH_TO_ROOT]
        assert config_const.CONF_DATASET_NAME in params, f"[data.holder_data_set] not configured"
        data_set = params[config_const.CONF_DATASET_NAME]
        data_set_processor = DataSetProcessor(params=params)
        folder_name = DataSetFactory.getDirectoryName(params)    
        root = os.path.join(root, folder_name)
        train_dataset, test_dataset, validation_dataset =  DataSetFactory().getDataSet(root, data_set, params)
        data_holder =  DataHolder(params=params, train_dataset=train_dataset, test_dataset=test_dataset, validation_dataset=validation_dataset)
        data_set_processor.initializeProcessors(data_holder)    
        return data_holder
    
    @staticmethod    
    def getDirectoryName(params):
        is_biderected, is_selfloop, is_not_multi, processor, transformer_model, process_id, is_normalized, is_sent_emb, edge_emb, conv_rel_2_node, only_amr = "",  "", "", "", "", "", "", "", "", "", ""
             
        if "graph.builder.bidirected" in params and params["graph.builder.bidirected"]:
            is_biderected = "_bidir"
        
        if "graph.builder.selfloop" in params and params["graph.builder.selfloop"]:
            is_selfloop = "_selfloop"
        
        if "graph.builder.multigraph" in params and not params["graph.builder.multigraph"]:
            is_not_multi = "_not_multi"            
        
        if "graph.builder.processor" in params and params["graph.builder.processor"]:
            processor = "_"+params["graph.builder.processor"]
        
        if "vector.embedding.transformer_model" in params and params["vector.embedding.transformer_model"]:
            transformer_model = "_"+params["vector.embedding.transformer_model"]
            
        if config_const.CONF_DATASET_PROCESS_ID in params and params[config_const.CONF_DATASET_PROCESS_ID]:
            process_id = "_"+params[config_const.CONF_DATASET_PROCESS_ID]
            
        if config_const.CONF_GRAPH_BUILDER_NORMALIZE_FEATURES in params and params[config_const.CONF_GRAPH_BUILDER_NORMALIZE_FEATURES]:
            is_normalized = "_norm"
        
        if config_const.CONF_EMBEDDING_SENTENCE_TRANSFORMER_MODEL in params and params[config_const.CONF_EMBEDDING_SENTENCE_TRANSFORMER_MODEL]:
            is_sent_emb = "_" + params[config_const.CONF_EMBEDDING_SENTENCE_TRANSFORMER_MODEL]
        
        if config_const.CONF_GRAPH_BUILDER_RELATION_FROM_LM in params and params[config_const.CONF_GRAPH_BUILDER_RELATION_FROM_LM]:     
            edge_emb = "_edg_emb" 
        
        if config_const.CONF_GRAPH_BUILDER_RELATION_TO_NODE in params and params[config_const.CONF_GRAPH_BUILDER_RELATION_TO_NODE]:     
            conv_rel_2_node = "_rel2node"
             
        if config_const.CONF_GRAPH_ONLY_ARM in params and params[config_const.CONF_GRAPH_ONLY_ARM]:     
            only_amr = "_amr"
             
        return params[config_const.CONF_DATASET_NAME]+process_id+processor+is_biderected+is_selfloop+is_not_multi+is_normalized+edge_emb+is_sent_emb+conv_rel_2_node+only_amr+"_"+params["vector.embedding_name"]+transformer_model+"_"+"_".join(params[config_const.CONF_GRAPH_BUILDER_NAME])
           