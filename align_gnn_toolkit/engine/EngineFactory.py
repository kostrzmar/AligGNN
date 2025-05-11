from utils import ConfigUtils, config_const
from engine import AbstractEngine
from engine.impl import ExperimentEngineFactory, FineTuneEngineFactory
from tqdm import tqdm
from multiprocessing import Process
import logging
from datetime import datetime
import argparse

class EngineFactory:
    
    def getEngineType(self, config_utils) -> AbstractEngine: 
        mode = config_utils.getValue(config_const.CONF_SEC_RAY, config_const.CONF_RAY_ENABLE)
        engine_factory = None 
        if mode == str(False) or mode == False:
            engine_factory = ExperimentEngineFactory()
        elif mode == str(True) or mode == True:
            engine_factory = FineTuneEngineFactory()
        assert engine_factory, "Engine Factory unknown"
        
        engine_job = engine_factory.getEngine()
        engine_job.initialize(config_utils)
        return engine_job    
    
    def getConfigurationUtils(self, configurationFile, config_as_dict=None):
        if not config_as_dict:
            assert configurationFile ,"Configuration is empty"
            if isinstance(configurationFile, argparse.Namespace):
                return ConfigUtils(configurationFile.conf, path_to_template_file=configurationFile.temp)
            else:
                return ConfigUtils(configurationFile)
        else:
            return ConfigUtils(None, config_as_dict=config_as_dict)
    
    def getUniqueProcessingId(self):
        return datetime.now().isoformat(timespec='milliseconds')
       
    
    def startExecution(self, configurationFile):
        
        config_utils = self.getConfigurationUtils(configurationFile)
        config_utils.setValue(config_const.CONF_SEC_EXPERIMENT, config_const.CONF_EXPERIMENT_UNIQUE_PROCESSING_ID, self.getUniqueProcessingId())
        config_utils.setValue(config_const.CONF_SEC_EXPERIMENT, config_const.CONF_EXPERIMENT_CONFIGURATION_FILE, configurationFile.conf)
        
        if config_utils.hasMultipleConfiguration():

            nbr_of_configurations = config_utils.getNumberOfConfiguration()
            nbr_of_processes = config_utils.getValue(config_const.CONF_SEC_EXPERIMENT, config_const.CONF_EXPERIMENT_NBR_PROCESSES, defaultValue=1)
            pbar = tqdm(total=nbr_of_configurations, desc="Experiments")
            conf_to_process = [[i for i in range(config_utils.getNumberOfConfiguration())][i * nbr_of_processes:(i+1) *nbr_of_processes] for i in range((config_utils.getNumberOfConfiguration()+nbr_of_processes-1)//nbr_of_processes)]
            
                
            for conf_nbr in conf_to_process:
                pool = []
                for index in range(nbr_of_processes):
                    logging.info(f'----------------------------------------')
                    logging.info(f'Executing configuration {conf_nbr[index]+1} out of total {config_utils.getNumberOfConfiguration()}')
                    config_utils.setActiveConfiguration(conf_nbr[index])
                    engine_type = self.getEngineType(config_utils)
                    if nbr_of_processes>1:
                        p = Process(target=engine_type.execute, args=())
                        pool.append(p)
                        p.start()
                    else:
                        self.getEngineType(config_utils).execute()
                
                for p in pool:
                    p.join()
                    pbar.update() 
            
            pbar.close()
        else:
            self.getEngineType(config_utils).execute()

        
        
        
