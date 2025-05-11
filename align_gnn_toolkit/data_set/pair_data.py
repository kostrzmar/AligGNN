from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

class PairData(Data):
    def __init__(self, 
                 x_s=None, edge_index_s=None, edge_attr_s=None, node_labels_s=None, sent_embedding_s=None, 
                 x_t=None, edge_index_t=None, edge_attr_t=None, node_labels_t=None, sent_embedding_t=None,
                 y=None):
        super().__init__()
        self.x_s = x_s
        self.node_labels_s = node_labels_s
        self.edge_index_s = edge_index_s
        self.edge_attr_s = edge_attr_s
        self.sent_embedding_s = sent_embedding_s

        self.x_t = x_t   
        self.node_labels_t = node_labels_t 
        self.edge_index_t = edge_index_t
        self.edge_attr_t = edge_attr_t
        self.sent_embedding_t = sent_embedding_t            
        self.y = y
        
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)
    
    def get_token_by_index(self, index, vocab=None, ignore_unk=False, remove_roberta=True):
        out = ""
        if vocab:
            if len(index) ==1:
                if ignore_unk:
                    if index[0] == vocab.get_default_index():
                        return ""
                out= vocab.get_itos()[index[0]]
            else:

                out = "".join([vocab.get_itos()[x] if x!= vocab.get_default_index() and x!=-1 else "" for x in index ])
        else:
            if len(index) ==1:
                out= index
            else:
                out= "".join([index for x in index ])
        if remove_roberta:
            out = out.replace('Ġ', '')
        return out
        
    
    def get_sentences(self, vocab=None, ignore_unk=False, remove_roberta=False, include_index=False):
        source, target = "",""
        if vocab:
            def _convert_itos(item):   
                out = ""
                if len(item) ==1:
                    if ignore_unk:
                        if item[0] == vocab.get_default_index():
                            out= ""
                    out= vocab.get_itos()[item[0]]
                else:
                    out= "".join([vocab.get_itos()[x] if x!= vocab.get_default_index() and x!=-1 else "" for x in item ])
                
                if remove_roberta:
                    out = out.replace('Ġ', '')
                return out
            if hasattr(self, 'x_s_batch'):
                source = []
                target = []
                batched_node_labels, mask = to_dense_batch(self.node_labels_s, self.x_s_batch)
                for i, node_labels in enumerate(batched_node_labels):
                    if include_index:
                        source.append(" ".join([ f'[{index}][{x}][{_convert_itos(x)}]' for index, x in enumerate(self.node_labels_s)]).strip())
                    else:
                        source.append(" ".join([ _convert_itos(x) for x in node_labels[mask[i]]]).strip())

                batched_node_labels, mask = to_dense_batch(self.node_labels_t, self.x_t_batch)
                for i, node_labels in enumerate(batched_node_labels):
                    if include_index:
                        target.append(" ".join([ f'[{index}][{x}][{_convert_itos(x)}]' for index, x in enumerate(self.node_labels_t)]).strip())
                    else:    
                        target.append(" ".join([ _convert_itos(x) for x in node_labels[mask[i]]]).strip())

                
            else:
                if include_index:
                    source=" ".join([ f'[{index}][{x}][{_convert_itos(x)}]' for index, x in enumerate(self.node_labels_s)]).strip()
                    target=" ".join([ f'[{index}][{x}][{_convert_itos(x)}]' for index, x in enumerate(self.node_labels_t)]).strip()
                else:
                    source=" ".join([ _convert_itos(x) for x in self.node_labels_s]).strip()
                    target=" ".join([ _convert_itos(x) for x in self.node_labels_t]).strip()

        else:
            def _getItem(item):
                return item

            source=[ _getItem(x) for x in self.node_labels_s]
            target=[ _getItem(x) for x in self.node_labels_t]
        return source, target
     
    def print_sentences(self, vocab=None):
        source, target = self.get_sentences(vocab=vocab)
        print(f'Source: {source}')
        print(f'Target: {target}')
            
    def get_source(self):
        sent_embedding = None
        if "sent_embedding_s" in self.keys():
            sent_embedding = self.sent_embedding_s
            
        return Data(x=self.x_s,
                    edge_index=self.edge_index_s,
                    node_labels=self.node_labels_s,
                    edge_attr = self.edge_attr_s,
                    sent_embedding = sent_embedding,
                    y=self.y)
    
    def get_target(self):
        sent_embedding = None
        if "sent_embedding_t" in self.keys():
            sent_embedding = self.sent_embedding_t
        return Data(x=self.x_t,
                    edge_index=self.edge_index_t,
                    node_labels=self.node_labels_t,
                    edge_attr = self.edge_attr_t,
                    sent_embedding = sent_embedding,
                    y=self.y)
        
        
