from torch_geometric.loader import DataLoader
from utils import config_const

class DataHolder():
    def __init__(self, 
                 data_set=None, 
                 params=None,
                 transform=None,
                 train_dataset = None,
                 test_dataset = None,
                 validation_dataset = None
                 ) -> None:
        self.train_data_loader = None
        self.test_data_loader = None
        self.validation_data_loader = None
        if not data_set and params:
            assert config_const.CONF_DATASET_NAME in params, f"[data.holder_data_set] not configured"
            data_set = params[config_const.CONF_DATASET_NAME]
            
        self.initializeDataLoaders( 
                                  params =params,
                                  transform=transform,
                                  train_dataset = train_dataset,
                                  test_dataset = test_dataset,
                                  validation_dataset = validation_dataset
                                )

              
    
    def initializeDataLoaders(self,  
                            params,
                            transform = None,
                            train_dataset = None,
                            test_dataset = None,
                            validation_dataset = None
                            ):
            

    
        exclude_keys = None
        if "data.holder_exclude_keys" in params:
            exclude_keys = params["data.holder_exclude_keys"]
        
        if train_dataset:
            self.train_data_loader = DataLoader(train_dataset, 
                            batch_size=params[config_const.CONF_DATASET_BATCH_SIZE], 
                            shuffle=True,
                            follow_batch=['x_s', 'x_t'],
                            exclude_keys=exclude_keys
                            )
        if test_dataset:
            self.test_data_loader = DataLoader(test_dataset, 
                            batch_size=params[config_const.CONF_DATASET_BATCH_SIZE], 
                            shuffle=True,
                            follow_batch=['x_s', 'x_t'], 
                            exclude_keys=exclude_keys
                            )  
        if validation_dataset:
            self.validation_data_loader = DataLoader(validation_dataset, 
                            batch_size=params[config_const.CONF_DATASET_BATCH_SIZE], 
                            shuffle=True,
                            follow_batch=['x_s', 'x_t'], 
                            exclude_keys=exclude_keys
                            )    

    @property
    def train_data_set(self):
        return self.train_data_loader.dataset
    
    @property
    def test_data_set(self):
        return self.test_data_loader.dataset
    
    @property
    def validation_data_set(self):
        return self.validation_data_loader.dataset
    




        
