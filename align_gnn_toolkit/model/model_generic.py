from model import AbstractModel


class GenericModel(AbstractModel):
    def __init__(self, params, number_of_labels, edge_feature_size=None, vocab=None, embed_one_hot=None):
        super(GenericModel, self).__init__(params, number_of_labels, edge_feature_size, vocab, embed_one_hot)
        self.local_reset_parameters()

    def local_reset_parameters(self):
        pass
                
    def forward(self, data):
        
        if self.cross_conv:
            embedding_1, embedding_2= self.cross_convolution_pass(data)
        else:    
            embedding_1 = self.convolutional_pass(data.edge_index_s, data.x_s, data.edge_attr_s, data.x_s_batch)
            embedding_2 = self.convolutional_pass(data.edge_index_t, data.x_t, data.edge_attr_t, data.x_t_batch)

        size = data.x_s_batch[-1].item() + 1 
        embedding_1, embedding_2 = self.simple_read_out(embedding_1, data.x_s_batch, embedding_2, data.x_t_batch, size)
        
        scores = self.concat_embeddings(embedding_1, embedding_2)
        score = self.scoring_pass(scores)
        return score, embedding_1, embedding_2, 0, 0