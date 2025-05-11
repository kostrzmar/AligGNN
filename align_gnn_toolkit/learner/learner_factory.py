from learner.learner_GCN import LearnerGCN
from learner.learner_generic import LearnerGeneric




class LearnerFactory():
    
    @staticmethod
    def getLearner(learner, params=None, mlflow=None):
        try:
            if learner == "GCN":
                return LearnerGCN(params, mlflow)
            elif learner == "Generic":
                return LearnerGeneric(params, mlflow)
            
            
            raise AssertionError(f'Unknown learner: {learner}')
        except AssertionError as e:
            print(e)