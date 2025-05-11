from datasets import Dataset
from data_set.data_set_document_abstract import DatasetDocumentAbstract
import os
import json    
import re   
from tqdm import tqdm
import torch
import logging   
from multiprocessing import Pool
from utils import config_const

class NewselaDocumentDataset(DatasetDocumentAbstract):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, type="train", params=None, graph_builder=None, data_set_processor=None):
        DATA_SET_NAME = "newsela_documents"   
        self.document_name = None
        self.document_level = None
        self.document_names = None
        super().__init__(root, transform, pre_transform, pre_filter, data_set=DATA_SET_NAME, type=type, params=params, graph_builder=graph_builder, data_set_processor=data_set_processor)        
    
    
    def initialize_data_set_parameters(self):
        
        self.corpus_score_labels = {"notAligned":0,
                                    "aligned":1,
                                    "partialAligned":0.8
        }
        self.corpus_attribute_name = {  NewselaDocumentDataset.SENTENCE_TOKEN_NAME:"text", 
                                        NewselaDocumentDataset.SENTENCE_TOKEN_KEY:"key", 
                                        NewselaDocumentDataset.SENTENCE_TOKEN_ALIGNMENT_KEY:"sentence_alignments", 
                                        NewselaDocumentDataset.PARAGRAPH_TOKEN_ALIGNMENT_KEY:"paragraph_alignments",
                                        NewselaDocumentDataset.DOCUMENT_NAME:"document_name",
                                        NewselaDocumentDataset.DOCUMENT_LEVEL:"document_level", 
                                        NewselaDocumentDataset.PARAGRAPH_ID: "paragraph_id"
                                      
        }   
        
        self.data_set_type_name = {NewselaDocumentDataset.DATA_SET_TYPE_TRAIN:"train", 
                                   NewselaDocumentDataset.DATA_SET_TYPE_VALIDATION:"validation", 
                                   NewselaDocumentDataset.DATA_SET_TYPE_TEST:"test"  
        }
        
        self.path_to_data = self.params["newsela_document"]
        

        
        self.type_to_file = {#self.data_set_type_name[NewselaDocumentDataset.DATA_SET_TYPE_TRAIN]:"newsela-auto-all-data.json",   
                             self.data_set_type_name[NewselaDocumentDataset.DATA_SET_TYPE_TRAIN]:"newsela-auto-sample.json", 
                             self.data_set_type_name[NewselaDocumentDataset.DATA_SET_TYPE_TEST]:"newsela-auto-sample.json",  
                             self.data_set_type_name[NewselaDocumentDataset.DATA_SET_TYPE_VALIDATION]:"newsela-auto-sample.json"}      
    
    
    @property
    def processed_file_names(self):
        if self.expectedFile:
            if self.limit:
                return self.expectedFile[:self.limit]
            else:
                return self.expectedFile
        else:
            logging.info(f'Dataset:[{self.data_set}] type:[{self.type}] -> checking data consistency')
            expectedFile = []
            set = self.load_data_set()
            self.get_document_names(set)
                    
            size = len(self.document_names.keys())
            if self.limit:
                if size > self.limit:
                    size = self.limit
            for index, item in tqdm(enumerate(self.document_names), desc=self.type):
                for item_level in self.document_names[item]:
                    for paragraph_id in self.document_names[item][item_level]:
                        expectedFile.append(f'data_{item}_{item_level}_{paragraph_id}.pt')
            self.expectedFile = expectedFile
        return expectedFile
    
    def get_document_names(self, data_set):
        self.document_names = {}
        for item in data_set: 
            if item[NewselaDocumentDataset.DOCUMENT_NAME] not in self.document_names.keys():
                self.document_names[item[NewselaDocumentDataset.DOCUMENT_NAME]] = {}
                
            if  item[NewselaDocumentDataset.DOCUMENT_LEVEL] not in self.document_names[item[NewselaDocumentDataset.DOCUMENT_NAME]]:    
                self.document_names[item[NewselaDocumentDataset.DOCUMENT_NAME]][item[NewselaDocumentDataset.DOCUMENT_LEVEL]] = []
                
            if  item[NewselaDocumentDataset.PARAGRAPH_ID] not in self.document_names[item[NewselaDocumentDataset.DOCUMENT_NAME]][item[NewselaDocumentDataset.DOCUMENT_LEVEL]]:    
                self.document_names[item[NewselaDocumentDataset.DOCUMENT_NAME]][item[NewselaDocumentDataset.DOCUMENT_LEVEL]].append(item[NewselaDocumentDataset.PARAGRAPH_ID])
        
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{self.document_name}_{self.document_level}_{idx}.pt'))
        return data   
 
 
    def len(self):
        if self.document_level and self.document_name:
            return len(self.document_names[self.document_name][self.document_level])
        else:
            return len(self.processed_file_names)
 
    def get_data_set_column_names(self):
        return [self.corpus_attribute_name[NewselaDocumentDataset.SENTENCE_TOKEN_NAME]]   
    
    def load_dataset_by_type(self, type, limit=None):
        return self._load_data_set(type, limit)
    
    def _load_data_set(self, data_set_type, limit):
        file_name =  self.type_to_file[data_set_type]
        data = None
        with open(os.path.join(self.path_to_data,file_name), encoding='utf-8') as fh:
            data = json.load(fh)  
        
        articles_to_process = list(data.keys())
        if limit:
            articles_to_process=articles_to_process[:limit]       
        final_sentences = []
        final_sentences_keys = []
        final_sentence_alignments = []
        final_paragraph_alignments = []
        final_document_names= []
        final_document_levels= []
        final_paragraph_ids = [] 
        levels = ['0', '1', '2', '3', '4']
        for document_name in articles_to_process:
            for document_level in (x for x in levels if x in data[document_name].keys()): 
                sentence_keys = sorted(self.get_sentence_keys(data, document_name, document_level), key=lambda x: int("".join(filter(str.isdigit, x))))
                sentences = []
                for key in sentence_keys:
                    sentences.append(self.get_sentence(data, document_name, document_level, key))
                            
                sentence_alignments = self.get_alignments(sentence_keys, self.get_sentence_alignment(data, document_name, document_level))
                paragraph_alignments = self.get_alignments(sentence_keys, self.get_paragraph_alignment(data, document_name, document_level), is_sentence_alignment=False)
                document_names = ["-".join(x.split("-")[:-3])  for x in sentence_keys]
                document_levels = [x.split("-")[-3]  for x in sentence_keys]
                paragraph_ids = [x.split("-")[-2]  for x in sentence_keys]
                final_sentences.extend(sentences)
                final_sentences_keys.extend(sentence_keys)
                final_sentence_alignments.extend(sentence_alignments)
                final_paragraph_alignments.extend(paragraph_alignments)
                final_document_names.extend(document_names)
                final_document_levels.extend(document_levels)
                final_paragraph_ids.extend(paragraph_ids)
        

        return Dataset.from_dict({self.corpus_attribute_name[NewselaDocumentDataset.SENTENCE_TOKEN_NAME]: final_sentences, 
                                  self.corpus_attribute_name[NewselaDocumentDataset.SENTENCE_TOKEN_KEY]: final_sentences_keys, 
                                  self.corpus_attribute_name[NewselaDocumentDataset.SENTENCE_TOKEN_ALIGNMENT_KEY]: final_sentence_alignments, 
                                  self.corpus_attribute_name[NewselaDocumentDataset.PARAGRAPH_TOKEN_ALIGNMENT_KEY]: final_paragraph_alignments,
                                  self.corpus_attribute_name[NewselaDocumentDataset.DOCUMENT_NAME]:final_document_names,
                                  self.corpus_attribute_name[NewselaDocumentDataset.DOCUMENT_LEVEL]:final_document_levels, 
                                  self.corpus_attribute_name[NewselaDocumentDataset.PARAGRAPH_ID]:final_paragraph_ids, 
                                  })
    
    def get_alignments(self, sentence_keys, alignments, is_sentence_alignment=True):
        alignments_dic = {}
        sentence_alignments = sentence_keys.copy()
        
        for item in alignments:
            if item[0] not in alignments_dic.keys():
                alignments_dic[item[0]] = []
            alignments_dic[item[0]].append(item[1])
                
        
        for index, item in enumerate(sentence_alignments):
            if not is_sentence_alignment:
                item = "-".join(item.split("-")[:-1])
                
            if item in alignments_dic.keys():
                sentence_alignments[index] =  alignments_dic[item]
            else:
                sentence_alignments[index] = None
                
        return sentence_alignments
        
    
    def get_sections_from_key(self, key):
        return re.findall('\d+', key)
                
    def get_document_keys(self, data):
        return list(data.keys())     
            
    def get_sentence_keys(self, data, article, level):
        return list(data[article][level].keys())
            
    def get_sentence(self, data, article, level, key):
        return data[article][level][key]
    
    def get_sentence_alignment(self, data, article, level):
        return [x for x in data[article]['sentence_alignment'] if x[0].split('-')[-3] ==level ] 
    
    def get_paragraph_alignment(self, data, article, level):
        return [x for x in data[article]['paragraph_alignment'] if x[0].split('-')[-2] ==level ] 
    
    def get_paragraph_number(self, sentence_key):
        return sentence_key.split("-")[-2]
    
    def _generateDataLocal(self, set, type):
        document_name, document_level, sentence_key = None, None, None
        document_graph = None
        paragraph_number = None
        paragraph_document_level = None
        total_nbr_of_sentence = len(set)
        for id_index, item in tqdm(enumerate(set), total=len(set), desc=f'Processing: {type}'):
            document_name= item[self.corpus_attribute_name[DatasetDocumentAbstract.DOCUMENT_NAME]]
            document_level= item[self.corpus_attribute_name[DatasetDocumentAbstract.DOCUMENT_LEVEL]]
            sentence_key = item[self.corpus_attribute_name[DatasetDocumentAbstract.SENTENCE_TOKEN_KEY]]
            sentence_alignment = item[self.corpus_attribute_name[DatasetDocumentAbstract.SENTENCE_TOKEN_ALIGNMENT_KEY]]
            paragraph_alignment = item[self.corpus_attribute_name[DatasetDocumentAbstract.PARAGRAPH_TOKEN_ALIGNMENT_KEY]]
            sentence_data = self._generateData(item)
            if document_graph:
                if paragraph_number != self.get_paragraph_number(sentence_key):
                    master_builder = self.graph_builder.getBuilderByName("Master")
                    offset = master_builder.offset_node_feats
                    index_of_masters = (document_graph.x[:, offset] == 1).nonzero().squeeze(1)
                    
                    document_hierarchy = self.graph_builder.getBuilderByName("documentHierarchy")
                    sent_node_feats,sent_node_labels,sent_edge_index, sent_edge_feats = [],[None]*len(document_graph.node_labels),[], []
                    
                    new_node_id = document_hierarchy.addNode(sent_node_feats,sent_node_labels,"paragraph")
                    idx = []
                    for item in index_of_masters.tolist():
                        idx.append([item, new_node_id])
                    idx.append([new_node_id, new_node_id])    
                    document_hierarchy.linkToNode(idx, sent_edge_index,sent_edge_feats, "paragraph")
                    paragraph_graph = self.graph_builder.get_graph_data(sent_node_feats, sent_edge_index, [[sent_node_labels[-1]]], sent_edge_feats, None)

                    document_graph = self.update_document_graph(document_graph, paragraph_graph)
                    torch.save(document_graph, os.path.join(self.processed_dir, f'data_{document_name}_{paragraph_document_level}_{paragraph_number}.pt'))
                    document_graph = sentence_data
                    paragraph_number = self.get_paragraph_number(sentence_key)
                    paragraph_document_level = document_level
                    if id_index==total_nbr_of_sentence-1:
                        torch.save(document_graph, os.path.join(self.processed_dir, f'data_{document_name}_{paragraph_document_level}_{paragraph_number}.pt'))
                            
                else:
                    document_graph = self.update_document_graph(document_graph, sentence_data, update_edge_index=True)
                    paragraph_number = self.get_paragraph_number(sentence_key)
                    paragraph_document_level = document_level
                    if id_index==total_nbr_of_sentence-1:
                        torch.save(document_graph, os.path.join(self.processed_dir, f'data_{document_name}_{paragraph_document_level}_{paragraph_number}.pt'))
                
            else:
                document_graph = sentence_data
                paragraph_number = self.get_paragraph_number(sentence_key)
                paragraph_document_level = document_level
    
    
    def generateData(self, set, type):
        data_per_article = {}
        for item in tqdm(set, total=len(set), desc=type):
             document_name= item[self.corpus_attribute_name[DatasetDocumentAbstract.DOCUMENT_NAME]]
             if document_name not in data_per_article:
                 data_per_article[document_name] = []
             data_per_article[document_name].append(item)

            
        
        idx = 0
        if config_const.CONF_DATASET_NBR_PROCESSES in self.params and int(self.params[config_const.CONF_DATASET_NBR_PROCESSES])>1: 
            #with get_context("spawn").Pool(processes=int(self.params["data.holder_nbr_processes"])) as pool:
            with Pool(processes=int(self.params[config_const.CONF_DATASET_NBR_PROCESSES])) as pool:
                for document_name in tqdm(data_per_article.keys(), total=len(data_per_article.keys()), desc=f'Nbr of doc {type}'):
                    pool.apply_async(self._generateDataLocal, args=(data_per_article[document_name], type ))

                pool.close()
                pool.join()    
                            
        else:
            for document_name in tqdm(data_per_article.keys(), total=len(data_per_article.keys()), desc=f'Nbr of doc {type}'):                
                self._generateDataLocal(data_per_article[document_name], type)
               


                        
    def update_document_graph(self, document_graph, graph, update_edge_index=False):
        
        if update_edge_index:
            graph.edge_index = torch.add(graph.edge_index , document_graph.x.shape[0])
        document_graph.x = torch.cat( (document_graph.x, graph.x), dim=0)
        document_graph.edge_index = torch.cat((document_graph.edge_index, graph.edge_index), dim=1)
        document_graph.edge_attr = torch.cat((document_graph.edge_attr, graph.edge_attr), dim=0)
        max_length = max(document_graph.node_labels.shape[1], graph.node_labels.shape[1])
        if document_graph.node_labels.shape[1]<max_length:
            document_graph.node_labels = torch.nn.functional.pad(document_graph.node_labels, (0,max_length -document_graph.node_labels.shape[1]), 'constant', -1)
        elif graph.node_labels.shape[1]<max_length:
            graph.node_labels = torch.nn.functional.pad(graph.node_labels, (0,max_length -graph.node_labels.shape[1]), 'constant', -1)  
        document_graph.node_labels = torch.cat((document_graph.node_labels, graph.node_labels), dim=0)
        return document_graph
 
            
