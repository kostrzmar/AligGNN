import torch
from torch.utils.data import Dataset
import json 
from data_set.pair_data import PairData  

class NewselaPairDataset(Dataset):
    def __init__(self, split_documents, paragraph_data_set, transform=None, target_transform=None, params=None):
        PATH_TO_FILE = params["newsela_document"]
            
        self.split_documents = split_documents
        self.data = []
        self.paragraph_data_set = paragraph_data_set
        f = open(PATH_TO_FILE)
        self.data_json = json.load(f) 
        alignements = {}
        for key in self.data_json.keys():
            alignements[key] =  self.data_json[key]["paragraph_alignment"]
            
        for item in split_documents:
            mapping = alignements[item]
            for id in mapping:
                source, target = id[0].split("."), id[1].split(".")
                self.data.append(self.get_article_pair(paragraph_data_set, source[0]+".en", source[1].split("-")[1], int(source[1].split("-")[2]), target[1].split("-")[1], int(target[1].split("-")[2]))) 
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __repr__(self):
        return f'"NewselaPairDataset({self.__len__()})"'
    
    def get_article(self, data_set, name, level, paragraph):
        data_set.document_level = level
        data_set.document_name = name
        return data_set[paragraph]

    def _generatePairData(self, source_data, target_data):
        pair_data = PairData(x_s=source_data.x, 
                            edge_index_s=source_data.edge_index, 
                            edge_attr_s=source_data.edge_attr,
                            node_labels_s=source_data.node_labels,
                            x_t=target_data.x, 
                            edge_index_t=target_data.edge_index, 
                            edge_attr_t=target_data.edge_attr,
                            node_labels_t=target_data.node_labels,
                            y=torch.tensor(0, dtype=torch.float))
        
        return pair_data

    def get_article_pair(self, paragraph_data_set, name, src_level, src_paragraph, trg_level, trg_paragraph):
        source_paragraph = self.get_article(paragraph_data_set, name, src_level, src_paragraph)
        target_paragraph = self.get_article(paragraph_data_set, name, trg_level, trg_paragraph)
        return self._generatePairData(source_paragraph, target_paragraph)
    
    def get_split(self, train_fraction, test_fraction):
        data = list(self.data_json.keys())
        train_set = data[:int((len(data)+1)*train_fraction)]
        test_set = data[int((len(data)+1)*train_fraction):int((len(data)+1)*(train_fraction+test_fraction))]
        val_set = data[int((len(data)+1)*(train_fraction+test_fraction)):]
        return train_set, test_set, val_set
