from engine import AbstractEngine
from utils import ConfigUtils, config_const
import logging
import ray
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.nevergrad import NevergradSearch
import nevergrad as ng
from datetime import datetime

import os
from data_set.data_holder import DataHolder
class FineTuneEngine(AbstractEngine):
    
    def initialize(self, config_utils : ConfigUtils) -> None:
        self.local_config_utils = config_utils
        logging.info("Experiment Engine initialization done")
        
    def doProcessing(self) -> None:
        nbr_cpu = self.local_config_utils.getValue(config_const.CONF_SEC_RAY, config_const.CONF_RAY_NBR_CPU, defaultValue=4)
        nbr_gpu = self.local_config_utils.getValue(config_const.CONF_SEC_RAY, config_const.CONF_RAY_NBR_GPU, defaultValue=1)
        DataHolder(params=self.getProcessingParameters())  #Initialize Data...
        
        ray.shutdown()  # Restart Ray defensively in case the ray connection is lost. 
        ray.init(num_cpus=nbr_cpu, num_gpus=nbr_gpu,log_to_driver=False)
        best_min_parameters, best_max_parameters = self.fineTune()
        ray.shutdown()
        logging.info("Fine tune  Engine do processing done")
        logging.info("Execute model with best parameters")
        assert not ray.is_initialized()
        best_min_parameters['ray.enable'] = False
        best_min_parameters["mlflow.experiment_name"] = self.local_config_utils.getValue(config_const.CONF_SEC_MLFLOW, config_const.CONF_MLFLOW_EXPERIMENT_NAME)
        self.executeExperiment(best_min_parameters)


    def fineTune(self):
        search_space = self.getProcessingParameters()
        search_space["mlflow.experiment_name"] = "RAY_"+self.local_config_utils.getValue(config_const.CONF_SEC_MLFLOW, config_const.CONF_MLFLOW_EXPERIMENT_NAME)
        search_space["dataset.path_to_root"] = self.local_config_utils.getValue(config_const.CONF_SEC_DATASET, config_const.CONF_DATA_PATH_TO_ROOT) 
        search_space["data.holder_batch_size"] = self.local_config_utils.getValue(config_const.CONF_SEC_DATASET, config_const.CONF_DATASET_BATCH_SIZE, ray_fine_tune=True)
        search_space["optimizer.learning_rate"] = self.local_config_utils.getValue(config_const.CONF_SEC_OPTIMIZER, config_const.CONF_OPTIMIZER_LEARNING_RATE, ray_fine_tune=True)
        search_space["optimizer.weight_decay"] = self.local_config_utils.getValue(config_const.CONF_SEC_OPTIMIZER, config_const.CONF_OPTIMIZER_WEIGHT_DECAY, ray_fine_tune=True)
        search_space["model.dropout_rate"] = self.local_config_utils.getValue(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_DROPOUT_RATE, ray_fine_tune=True)    
                  
        search_space["model.number_heads"] = self.local_config_utils.getValue(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_NUMBER_HEADS, ray_fine_tune=True)
        search_space["model.embedding_size"] = self.local_config_utils.getValue(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_EMBEDDING_SIZE, ray_fine_tune=True)
        search_space["model.output_dim"] = self.local_config_utils.getValue(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_OUTPUT_DIM, ray_fine_tune=True)             

        ray_search_type = self.local_config_utils.getValue(config_const.CONF_SEC_RAY, config_const.CONF_RAY_SEARCH_TYPE)
        ray_num_samples = self.local_config_utils.getValue(config_const.CONF_SEC_RAY, config_const.CONF_RAY_NBR_SAMPLES, defaultValue=20)
        ray_output_folder_path = "./.ray_results"
        ray_output_experiment_folder_name = "execute_experiment_"+ self.local_config_utils.getValue(config_const.CONF_SEC_EXPERIMENT, config_const.CONF_EXPERIMENT_UNIQUE_PROCESSING_ID)
        ng_search = None
        if ray_search_type and ray_search_type =='nevergrad':
            ng_search = NevergradSearch(
                optimizer=ng.optimizers.OnePlusOne,
                metric="min_loss",
                mode="min")
        
        tuner = tune.Tuner(

        self.executeExperiment,
        tune_config=tune.TuneConfig(
            num_samples=ray_num_samples,
            search_alg=ng_search,
            scheduler=ASHAScheduler(
                max_t=10,
                grace_period=1,
                reduction_factor=2,
                metric="min_loss", 
                mode="min"
                )
                ),
        param_space=search_space,
        run_config=air.RunConfig(local_dir=ray_output_folder_path, name=ray_output_experiment_folder_name)
        )
        results = tuner.fit()

        df = results.get_dataframe()
        print(f'Stats of min_loss:')
        print(df.min_loss.describe())
        df.min_loss.describe().to_csv(os.path.join(ray_output_folder_path, ray_output_experiment_folder_name, "results_loss_summary.csv"), index=False)
        df.to_csv(os.path.join(ray_output_folder_path, ray_output_experiment_folder_name, "results_output.csv"), index=False)
        
        uniqueProcessingId = self.local_config_utils.getValue(config_const.CONF_SEC_EXPERIMENT, config_const.CONF_EXPERIMENT_UNIQUE_PROCESSING_ID)
        path_to_root= self.local_config_utils.getValue(config_const.CONF_SEC_DATASET, config_const.CONF_DATA_PATH_TO_ROOT)
        path = os.path.join(path_to_root, self.getExperimentOutputPath(uniqueProcessingId, self.local_config_utils.getValue(config_const.CONF_SEC_EXPERIMENT, config_const.CONF_EXPERIMENT_PATH_TO_ROOT)))
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        output_file = os.path.join(path, "fine_tune_results_"+datetime.now().isoformat(timespec='seconds')+".csv")
        df.to_csv(output_file, index=False)
        
        return results.get_best_result(metric="min_loss", mode="min").config, results.get_best_result(metric="min_loss", mode="max").config
        
