
from learner import AbstractLearner
from model import GCN



class LearnerGCN(AbstractLearner):
    
    def __init__(self,
                params=None, 
                mlflow=None
                ) -> None:
        super(LearnerGCN, self).__init__(params, mlflow)

        
    def getLocalParameters(self):
        return {}
    
    def getModel(self):
        return GCN(feature_size=self.get_feature_size(), edge_feature_size=None, model_params=self.get_model_params())


 