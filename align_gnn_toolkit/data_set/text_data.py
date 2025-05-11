from torch_geometric.data import Data

class TextData(Data):
    def __init__(self, 
                 x=None, edge_index=None, edge_attr=None, node_labels=None,y=None):
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        self.node_labels = node_labels

    def get_token_by_index(self, index, vocab=None, ignore_unk=False):
        if vocab:
            if len(index) ==1:
                if ignore_unk:
                    if index[0] == vocab.get_default_index():
                        return ""
                return vocab.get_itos()[index[0]]
            else:

                return "".join([vocab.get_itos()[x] if x!= vocab.get_default_index() else "" for x in index ])
        else:
            if len(index) ==1:
                return index
            else:
                return "".join([index for x in index ])
        
    def _get_sentences(self, vocab=None, ignore_unk=False, labels=None):
        sentence= ""
        if vocab:
            def _convert_itos(item):
                if len(item) ==1:
                    if ignore_unk:
                        if item[0] == vocab.get_default_index():
                            return ""
                    return vocab.get_itos()[item[0]]
                else:
                    return "".join([vocab.get_itos()[x] if x!= vocab.get_default_index() and x!=-1 else "" for x in item ])
            sentence=" ".join([ _convert_itos(x) for x in labels]).strip()
        else:
            def _getItem(item):
                return item
            sentence=[ _getItem(x) for x in labels]

        return sentence
     
    def print_sentences(self, vocab=None):
        print(f'Source: {self._get_sentences(vocab=vocab, labels=self.node_labels)}')
            
    def get_source(self):
        return self
    
    def get_target(self):
        return None
        
        