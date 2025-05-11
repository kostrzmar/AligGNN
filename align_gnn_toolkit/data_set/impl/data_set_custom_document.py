from datasets import Dataset
from data_set.data_set_abstract import DatasetAbstract
from data_set.data_set_document_abstract import DatasetDocumentAbstract
import os
from tqdm import tqdm
import torch
import numpy as np

       
class CustomDocumentDataset(DatasetDocumentAbstract):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, type="train", params=None, graph_builder=None, data_set_processor=None):
        DATA_SET_NAME = "custom_data_set"  
        super().__init__(root, transform, pre_transform, pre_filter, data_set=DATA_SET_NAME, type=type, params=params, graph_builder=graph_builder, data_set_processor=data_set_processor)        


    def initialize_data_set_parameters(self):
        self.corpus_score_normalization_const = 3
        self.corpus_score_labels = {"notAligned":0,
                                    "aligned":1,
                                    "partialAligned":0.8
        }
        self.corpus_attribute_name = {  CustomDocumentDataset.SOURCE+CustomDocumentDataset.SENTENCE_TOKEN_NAME:"src_text",   
                                        CustomDocumentDataset.TARGET+CustomDocumentDataset.SENTENCE_TOKEN_NAME:"trg_text",  
                                        CustomDocumentDataset.SCORE_TOKEN_NAME:"aligment"
        }   
        

        self.data_set_type_name = {CustomDocumentDataset.DATA_SET_TYPE_TRAIN:"train", 
                                CustomDocumentDataset.DATA_SET_TYPE_VALIDATION:"validation", 
                                CustomDocumentDataset.DATA_SET_TYPE_TEST:"test"
        }
        

    def get_document_names(self, data_set):
        self.document_names = {}
        self.document_names[self.data_set] = ['0']

    def load_dataset_by_type(self, type, limit=None):
        return self._load_data_set(type, limit)
    
    def _load_data_set(self, data_set_type, limit):
        org = self.params["data.custom_dataset_org"]
        org_paragraph = self.params["data.custom_dataset_org_paragraph"]
        trg = self.params["data.custom_dataset_trg"]
        trg_paragraph = self.params["data.custom_dataset_trg_paragraph"]
        scores = self.params["data.custom_dataset_scores"]
        labels = self.params["data.custom_dataset_labels"]
        dataset = Dataset.from_dict({labels['org']: org, labels['org_paragraph']: org_paragraph, labels['trg']: trg, labels['trg_paragraph']: trg_paragraph, labels['scr']: scores})
        
        return dataset


    def generateData(self, set, type):
        idx = 0     
        document_name, document_level, sentence_key = None, None, None
        document_graph = None
        for item in tqdm(set, total=len(set), desc=type):
            document_name= item[self.corpus_attribute_name[DatasetDocumentAbstract.DOCUMENT_NAME]]
            document_level= item[self.corpus_attribute_name[DatasetDocumentAbstract.DOCUMENT_LEVEL]]
            sentence_key = item[self.corpus_attribute_name[DatasetDocumentAbstract.SENTENCE_TOKEN_KEY]]
            sentence_alignment = item[self.corpus_attribute_name[DatasetDocumentAbstract.SENTENCE_TOKEN_ALIGNMENT_KEY]]
            paragraph_alignment = item[self.corpus_attribute_name[DatasetDocumentAbstract.PARAGRAPH_TOKEN_ALIGNMENT_KEY]]
            sentence_data = self._generateData(item)
            if document_graph:
                sentence_data.edge_index = torch.add(sentence_data.edge_index , document_graph.x.shape[0])
                document_graph.x = torch.cat( (document_graph.x, sentence_data.x), dim=0)
                document_graph.edge_index = torch.cat((document_graph.edge_index, sentence_data.edge_index), dim=1)
                document_graph.edge_attr = torch.cat((document_graph.edge_attr, sentence_data.edge_attr), dim=0)
                
                max_length = max(document_graph.node_labels.shape[1], sentence_data.node_labels.shape[1])
                if document_graph.node_labels.shape[1]<max_length:
                    document_graph.node_labels = torch.nn.functional.pad(document_graph.node_labels, (0,max_length -document_graph.node_labels.shape[1]), 'constant', -1)
                elif sentence_data.node_labels.shape[1]<max_length:
                    sentence_data.node_labels = torch.nn.functional.pad(sentence_data.node_labels, (0,max_length -sentence_data.node_labels.shape[1]), 'constant', -1)
                            
                document_graph.node_labels = torch.cat((document_graph.node_labels, sentence_data.node_labels), dim=0)
                
            else:
                document_graph = sentence_data
            
        torch.save(document_graph, os.path.join(self.processed_dir, f'data_{document_name}_{document_level}_{idx}.pt'))
        idx += 1  
