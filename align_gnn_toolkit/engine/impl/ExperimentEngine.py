
from engine import AbstractEngine
from utils import ConfigUtils, config_const
import logging

class ExperimentEngine(AbstractEngine):
    
    def initialize(self, config_utils : ConfigUtils) -> None:     
        self.local_config_utils = config_utils
        logging.info("Experiment Engine initialization done")

        
    def doProcessing(self) -> None:
        do_fine_tuning = self.local_config_utils.getValue(config_const.CONF_SEC_EXPERIMENT, config_const.CONF_DO_FINE_TUNING, defaultValue=False)
        if do_fine_tuning:
            fine_tuning_def = self.local_config_utils.getValue(config_const.CONF_SEC_EXPERIMENT, config_const.CONF_FINE_TUNING_DEF)
            processing_parameters = self.getProcessingParameters()
            nbr_of_fine_tune_trails =  len(fine_tuning_def[list(fine_tuning_def.keys())[0]])
            logging.info(f'Fine tune of {list(fine_tuning_def.keys())} in [{nbr_of_fine_tune_trails}] trails')
            for index in range(nbr_of_fine_tune_trails):
                for key in fine_tuning_def.keys():
                    processing_parameters[key] = fine_tuning_def[key][index]
                self.executeExperiment(processing_parameters)
        else:
            self.executeExperiment(self.getProcessingParameters())
        logging.info("Experiment Engine do processing done")
        
