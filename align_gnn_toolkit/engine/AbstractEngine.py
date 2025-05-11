from abc import ABC, abstractmethod
from utils import ConfigUtils,config_const
import numpy as np
import os
from learner import LearnerFactory
import utils.executor_utils as utils_processing
from torch_geometric.loader import DataLoader
import mlflow.pytorch
from ray import tune
import logging
import csv
import torch
import configparser
import pandas as pd
import time
from utils import executor_utils as utils
from data_set import DataSetFactory
import shutil
from utils import config_const
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
import utils.alignment_func as alig_func
from sklearn import metrics



PATH_TO_ENV_CONFIG = '../conf/config.cfg'
MLFLOW_URL="http://localhost:5000"

class AbstractEngine(ABC):


    @abstractmethod
    def initialize(self, config_utils : ConfigUtils) -> None:
        pass
       
    @abstractmethod
    def doProcessing(self) -> None:
        pass

    def getExperimentOutputPath(self, uniqueProcessingId, params):
        root = "./experiments"
        if config_const.CONF_EXPERIMENT_PATH_TO_ROOT in params and params[config_const.CONF_EXPERIMENT_PATH_TO_ROOT]:
            root = params[config_const.CONF_EXPERIMENT_PATH_TO_ROOT]
        return os.path.join(root, uniqueProcessingId)

    def makeOutputFolder(self, uniqueProcessingId, params):
        path = self.getExperimentOutputPath(uniqueProcessingId, params)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def __init__(self):
        self.local_config_utils = None
        self.output_folder_for_process = None
        self.is_reverse_processing = False
        super().__init__()


    def getProcessingParameters(self):
        def _get_values(section, property):
            return self.local_config_utils.getValue(section, property)
            
        hyper_parameters ={   
                "ray.enable"                : _get_values(config_const.CONF_SEC_RAY, config_const.CONF_RAY_ENABLE),
                "mlflow.enable"             : _get_values(config_const.CONF_SEC_MLFLOW, config_const.CONF_MLFLOW_ENABLE),
                "mlflow.experiment_name"    : _get_values(config_const.CONF_SEC_MLFLOW, config_const.CONF_MLFLOW_EXPERIMENT_NAME),
                "learner.name"              : _get_values(config_const.CONF_SEC_LEARNER, config_const.CONF_LEARNER_NAME),
                "vector.embedding_lang"     : self.local_config_utils.getValue(config_const.CONF_SEC_EMBEDDING, config_const.CONF_EMBEDDING_LANG,
                                                                               self.local_config_utils.getValue(config_const.CONF_SEC_EMBEDDING, "vector.embedding_lang")),
                "vector.embedding_name"     : self.local_config_utils.getValue(config_const.CONF_SEC_EMBEDDING, config_const.CONF_EMBEDDING_NAME,
                                                                               self.local_config_utils.getValue(config_const.CONF_SEC_EMBEDDING, "vector.embedding_name")),
                "vector.embedding.transformer_model" : self.local_config_utils.getValue(config_const.CONF_SEC_EMBEDDING, config_const.CONF_EMBEDDING_TRANSFORMER_MODEL,
                                                                                        self.local_config_utils.getValue(config_const.CONF_SEC_EMBEDDING, "vector.embedding.transformer_model")),
                "graph.builder.processor"   : _get_values(config_const.CONF_SEC_GRAPH, config_const.CONF_GRAPH_BUILDER_PROCESSOR),
                "graph.builder.selfloop"    : _get_values(config_const.CONF_SEC_GRAPH, config_const.CONF_GRAPH_BUILDER_SELFLOOP),
                "graph.builder.bidirected"  : _get_values(config_const.CONF_SEC_GRAPH, config_const.CONF_GRAPH_BUILDER_BIDIRECTED), 
                "data.holder_exclude_keys"  : ["node_labels_s", "node_labels_t"],
                "optimizer.name"            : _get_values(config_const.CONF_SEC_OPTIMIZER, config_const.CONF_OPTIMIZER_NAME),
                "optimizer.learning_rate"   : _get_values(config_const.CONF_SEC_OPTIMIZER, config_const.CONF_OPTIMIZER_LEARNING_RATE),
                "optimizer.weight_decay"    : _get_values(config_const.CONF_SEC_OPTIMIZER, config_const.CONF_OPTIMIZER_WEIGHT_DECAY),
                "loss_function.name"        : _get_values(config_const.CONF_SEC_TRAIN, config_const.CONF_TRAIN_LOSS_FUNCTION_NAME),
                "loss_function.coefficients": _get_values(config_const.CONF_SEC_TRAIN, config_const.CONF_TRAIN_LOSS_COEFFICIENTS),
                "train.epoch_number"        : _get_values(config_const.CONF_SEC_TRAIN, config_const.CONF_TRAIN_EPOCH_NUMBER),
                "train.early_stop_by"       : _get_values(config_const.CONF_SEC_TRAIN, config_const.CONF_TRAIN_EARLY_STOP_BY),
                "train.show_info_by"        : _get_values(config_const.CONF_SEC_TRAIN, config_const.CONF_TRAIN_SHOW_INFO_BY),
                "model.number_heads"        : _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_NUMBER_HEADS),
                "model.embedding_size"      : _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_EMBEDDING_SIZE),
                "model.output_dim"          : _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_OUTPUT_DIM),
                "model.dropout_rate"        : _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_DROPOUT_RATE),      
                "model.similarity_score_type"     : _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_SIMILARITY_SCORE_TYPE),
                "model.similarity_score_norm_type" : _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_SIMILARITY_SCORE_NORM_TYPE),                
                "data.custom_dataset_org": _get_values(config_const.CONF_SEC_DATASET, "data.custom_dataset_org"),
                "data.custom_dataset_trg": _get_values(config_const.CONF_SEC_DATASET, "data.custom_dataset_trg"), 
                "data.custom_dataset_scores": _get_values(config_const.CONF_SEC_DATASET, "data.custom_dataset_scores"),
                "data.custom_dataset_labels": _get_values(config_const.CONF_SEC_DATASET, "data.custom_dataset_labels"),
                                                
                config_const.CONF_MLFLOW_EXPERIMENT_NAME_GENERATE: _get_values(config_const.CONF_SEC_MLFLOW, config_const.CONF_MLFLOW_EXPERIMENT_NAME_GENERATE),
                config_const.CONF_EMBEDDING_SENTENCE_TRANSFORMER_MODEL : _get_values(config_const.CONF_SEC_EMBEDDING, config_const.CONF_EMBEDDING_SENTENCE_TRANSFORMER_MODEL),
                config_const.CONF_EXPERIMENT_IGNORE_VALUATION: _get_values(config_const.CONF_SEC_EXPERIMENT,config_const.CONF_EXPERIMENT_IGNORE_VALUATION),                
                config_const.CONF_GRAPH_BUILDER_NAME        : self.local_config_utils.getValue(config_const.CONF_SEC_GRAPH, config_const.CONF_GRAPH_BUILDER_NAME,
                                                                               self.local_config_utils.getValue(config_const.CONF_SEC_GRAPH, "graph.builder_name")),
                config_const.CONF_GRAPH_BUILDER_NORMALIZE_FEATURES : _get_values(config_const.CONF_SEC_DATASET, config_const.CONF_GRAPH_BUILDER_NORMALIZE_FEATURES),
                config_const.CONF_GRAPH_BUILDER_MULTIGRAPH :    _get_values(config_const.CONF_SEC_GRAPH, config_const.CONF_GRAPH_BUILDER_MULTIGRAPH),
                config_const.CONF_GRAPH_BUILDER_RELATION_FROM_LM :_get_values(config_const.CONF_SEC_GRAPH, config_const.CONF_GRAPH_BUILDER_RELATION_FROM_LM),
                config_const.CONF_GRAPH_BUILDER_RELATION_TO_NODE :_get_values(config_const.CONF_SEC_GRAPH, config_const.CONF_GRAPH_BUILDER_RELATION_TO_NODE),
                config_const.CONF_GRAPH_ONLY_ARM :_get_values(config_const.CONF_SEC_GRAPH, config_const.CONF_GRAPH_ONLY_ARM),
                config_const.CONF_DATASET_NAME : self.local_config_utils.getValue(config_const.CONF_SEC_DATASET, config_const.CONF_DATASET_NAME, 
                                                                                 self.local_config_utils.getValue(config_const.CONF_SEC_DATASET, "data.holder_data_set")),
                config_const.CONF_DATASET_BATCH_SIZE : self.local_config_utils.getValue(config_const.CONF_SEC_DATASET, config_const.CONF_DATASET_BATCH_SIZE),
                config_const.CONF_DATASET_LIMIT : self.local_config_utils.getValue(config_const.CONF_SEC_DATASET, config_const.CONF_DATASET_LIMIT,
                                                                                   self.local_config_utils.getValue(config_const.CONF_SEC_DATASET, "data.holder_dataset_limit")),
                config_const.CONF_DATASET_REGENERATE_GRAPH : _get_values(config_const.CONF_SEC_DATASET, config_const.CONF_DATASET_REGENERATE_GRAPH),
                config_const.CONF_CURRICULUM_LEARNING : _get_values(config_const.CONF_SEC_TRAIN, config_const.CONF_CURRICULUM_LEARNING),
                config_const.CONF_TRAIN_LOGITS_CONVERTER: _get_values(config_const.CONF_SEC_TRAIN, config_const.CONF_TRAIN_LOGITS_CONVERTER),
                config_const.CONF_DATASET_USE_CACHE : _get_values(config_const.CONF_SEC_DATASET, config_const.CONF_DATASET_USE_CACHE),
                config_const.CONF_DATASET_ID_FROM : _get_values(config_const.CONF_SEC_DATASET, config_const.CONF_DATASET_ID_FROM),
                config_const.CONF_DATASET_ID_TO : _get_values(config_const.CONF_SEC_DATASET, config_const.CONF_DATASET_ID_TO),
                config_const.CONF_DATASET_PROCESS_PARALLEL : _get_values(config_const.CONF_SEC_DATASET, config_const.CONF_DATASET_PROCESS_PARALLEL),
                config_const.CONF_DATASET_NBR_PROCESSES : self.local_config_utils.getValue(config_const.CONF_SEC_DATASET, config_const.CONF_DATASET_NBR_PROCESSES, 
                                                                                           self.local_config_utils.getValue(config_const.CONF_SEC_DATASET,"data.holder_nbr_processes",1)),          
                config_const.CONF_DATASET_INIT_SINGLE_DATASET : _get_values(config_const.CONF_SEC_DATASET, config_const.CONF_DATASET_INIT_SINGLE_DATASET),
                config_const.CONF_DATASET_PATH_TO_ROOT : _get_values(config_const.CONF_SEC_DATASET, config_const.CONF_DATASET_PATH_TO_ROOT),
                config_const.CONF_DATASET_IN_MEMORY_ONLY : _get_values(config_const.CONF_SEC_DATASET, config_const.CONF_DATASET_IN_MEMORY_ONLY),
                config_const.CONF_DATASET_PROCESS_ID: _get_values(config_const.CONF_SEC_DATASET, config_const.CONF_DATASET_PROCESS_ID),
                config_const.CONF_DATASET_BINARIZE:  _get_values(config_const.CONF_SEC_DATASET, config_const.CONF_DATASET_BINARIZE),
                config_const.CONF_MODEL_LOCK_CONV_LAYERS: _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_LOCK_CONV_LAYERS),
                config_const.CONF_MODEL_LOCK_CONV_LAYERS_AFTER_EPOCH: _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_LOCK_CONV_LAYERS_AFTER_EPOCH),
                config_const.CONF_OPTIMIZER_LEARNING_RATE_PATIENCE: _get_values(config_const.CONF_SEC_OPTIMIZER, config_const.CONF_OPTIMIZER_LEARNING_RATE_PATIENCE),
                config_const.CONF_SCHEDULER_NAME: _get_values(config_const.CONF_SEC_OPTIMIZER, config_const.CONF_SCHEDULER_NAME),
                config_const.CONF_TRAIN_SHOW_EPOCH_INFO_BY: _get_values(config_const.CONF_SEC_TRAIN, config_const.CONF_TRAIN_SHOW_EPOCH_INFO_BY),
                config_const.CONF_TRAIN_LOSS_ALPHA : _get_values(config_const.CONF_SEC_TRAIN, config_const.CONF_TRAIN_LOSS_ALPHA),
                config_const.CONF_TRAIN_LOSS_BETA : _get_values(config_const.CONF_SEC_TRAIN, config_const.CONF_TRAIN_LOSS_BETA),
                config_const.CONF_ENV_PATH  : _get_values(config_const.CONF_SEC_ENV, config_const.CONF_ENV_PATH),  
                config_const.CONF_MODEL_SIMGNN_GNN_OPERATOR : _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_SIMGNN_GNN_OPERATOR),
                config_const.CONF_MODEL_SIMGNN_FILTERS_1 : _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_SIMGNN_FILTERS_1), 
                config_const.CONF_MODEL_SIMGNN_FILTERS_2 : _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_SIMGNN_FILTERS_2),
                config_const.CONF_MODEL_SIMGNN_FILTERS_3 : _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_SIMGNN_FILTERS_3),
                config_const.CONF_MODEL_SIMGNN_TENSOR_NEURONS : _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_SIMGNN_TENSOR_NEURONS),
                config_const.CONF_MODEL_SIMGNN_BOTTLE_NECK_NEURONS : _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_SIMGNN_BOTTLE_NECK_NEURONS),
                config_const.CONF_MODEL_SIMGNN_BINS : _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_SIMGNN_BINS),
                config_const.CONF_MODEL_SIMGNN_HISTOGRAM : _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_SIMGNN_HISTOGRAM),
                config_const.CONF_MODEL_SIMGNN_DIFFPOOL : _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_SIMGNN_DIFFPOOL),
                config_const.CONF_MODEL_SOFT_MAX_AGGR: _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_SOFT_MAX_AGGR),
                config_const.CONF_MODEL_DROPOUT_RATE: _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_DROPOUT_RATE),
                config_const.CONF_MODEL_CONV_LAYERS_NBR: _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_CONV_LAYERS_NBR),
                config_const.CONF_MODEL_CONV_DO_SKIP_CONNECTION: _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_CONV_DO_SKIP_CONNECTION),
                config_const.CONF_MODEL_CONV_ACTIVATION_TYPE:_get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_CONV_ACTIVATION_TYPE),
                config_const.CONF_MODEL_SCORING_TYPE: _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_SCORING_TYPE),
                config_const.CONF_MODEL_READ_OUT_TYPE: _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_READ_OUT_TYPE)  ,          
                config_const.CONF_MODEL_STORE_BEST_MODEL_ON_METRIC: _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_STORE_BEST_MODEL_ON_METRIC),
                config_const.CONF_MODEL_EMBED_ONE_HOT: _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_EMBED_ONE_HOT),
                config_const.CONF_MODEL_ONE_HOT_EMBEDDING_SIZE: _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_ONE_HOT_EMBEDDING_SIZE),
                config_const.CONF_MODEL_POOL_GNN_OPERATOR: _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_POOL_GNN_OPERATOR),
                config_const.CONF_OPTIMIZER_LEARNING_WARMUP_STEPS: _get_values(config_const.CONF_SEC_OPTIMIZER, config_const.CONF_OPTIMIZER_LEARNING_WARMUP_STEPS),
                config_const.CONF_OPTIMIZER_LEARNING_WARMUP_INIT_LR: _get_values(config_const.CONF_SEC_OPTIMIZER, config_const.CONF_OPTIMIZER_LEARNING_WARMUP_INIT_LR),
                config_const.CONF_EXPERIMENT_VERIFY_ON_EVAL: _get_values(config_const.CONF_SEC_EXPERIMENT, config_const.CONF_EXPERIMENT_VERIFY_ON_EVAL),
                config_const.CONF_EXPERIMENT_VERIFY_ON_EVAL_DS: _get_values(config_const.CONF_SEC_EXPERIMENT, config_const.CONF_EXPERIMENT_VERIFY_ON_EVAL_DS),
                config_const.CONF_EXPERIMENT_VERIFY_ON_ALIGNMENT: _get_values(config_const.CONF_SEC_EXPERIMENT, config_const.CONF_EXPERIMENT_VERIFY_ON_ALIGNMENT),
                config_const.CONF_DO_FINE_TUNING: _get_values(config_const.CONF_SEC_EXPERIMENT, config_const.CONF_DO_FINE_TUNING),
                config_const.CONF_FINE_TUNING_DEF: _get_values(config_const.CONF_SEC_EXPERIMENT, config_const.CONF_FINE_TUNING_DEF),
                config_const.CONF_EXPERIMENT_CONFIGURATION_FILE: _get_values(config_const.CONF_SEC_EXPERIMENT, config_const.CONF_EXPERIMENT_CONFIGURATION_FILE),
                config_const.CONF_ALIGNMENT_GRAPH_DATA_SET: _get_values(config_const.CONF_SEC_ALIGNMENT, config_const.CONF_ALIGNMENT_GRAPH_DATA_SET),
                config_const.CONF_ALIGNMENT_SIMILARITY_METRIC: _get_values(config_const.CONF_SEC_ALIGNMENT, config_const.CONF_ALIGNMENT_SIMILARITY_METRIC),
                config_const.CONF_ALIGNMENT_MODEL: _get_values(config_const.CONF_SEC_ALIGNMENT, config_const.CONF_ALIGNMENT_MODEL),
                config_const.CONF_EXPERIMENT_PATH_TO_ROOT: _get_values(config_const.CONF_SEC_EXPERIMENT, config_const.CONF_EXPERIMENT_PATH_TO_ROOT),
                config_const.CONF_MODEL_PATH_TO_ROOT: _get_values(config_const.CONF_SEC_MODEL, config_const.CONF_MODEL_PATH_TO_ROOT),
            }
        
        out = {}
        out.update(hyper_parameters)
        out.update(self.get_env_configuration(hyper_parameters))
        
        return out    
    
    def get_env_configuration(self, hyper_parameters):
        env_cfg = {}
        if config_const.CONF_ENV_PATH in hyper_parameters and hyper_parameters[config_const.CONF_ENV_PATH]:
            cfg = configparser.ConfigParser()
            cfg.read_file(open(hyper_parameters[config_const.CONF_ENV_PATH]))
            for k in cfg['DEFAULT'].keys():
                env_cfg[k] = cfg['DEFAULT'][k]
        else:
            logging.error(f'Configuration for external datasets not provided. Check property {config_const.CONF_ENV_PATH}')
        return env_cfg


    def execute(self) -> None:
        self.doProcessing()
            
    def storeLearningResults(self, trial, out_statistic, experiment_params,  uniqueProcessingId, is_final_trial=False):
        if is_final_trial:
            logging.info(f'Final statistic for [{trial+1}] trials:')
            raw_figures = {}
            for key in out_statistic.keys():
                if isinstance(out_statistic[key], list) and utils.is_numeric(out_statistic[key][0]):
                    raw_figures[key+"_raw"] = f'[{out_statistic[key]}]'
                    out_statistic[key] = f'{np.average(out_statistic[key]):.4f} ± {np.std(out_statistic[key]):.4f}'
                    
                    logging.info(f'[{key}]: {out_statistic[key]}')
            out_statistic.update(raw_figures)
            out_statistic["Nbr of trials"] = trial+1
            trial = "final"
            if "trail_model_path" in out_statistic and isinstance(out_statistic["trail_model_path"], list):
                best_models = []
                for key in out_statistic.keys():
                    if key.endswith("_per_trail_model"):
                        best_models.append(out_statistic[key])
                to_delete = list(set(out_statistic["trail_model_path"])- set(best_models))
                for path_to_remove in to_delete:
                    logging.info(f'Deleting not performing models: [{path_to_remove}]')
                    shutil.rmtree(path_to_remove, ignore_errors=True)
                
            
        output_file = os.path.join(self.getExperimentOutputPath(uniqueProcessingId, experiment_params), "trail_"+str(trial)+"_learning_results.csv")
        add_header=False
        if not os.path.isfile(output_file):
            add_header = True
        out_statistic.update(experiment_params)
        out = sorted(out_statistic)
        with open(output_file, 'a+', encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";", escapechar="/", quoting=csv.QUOTE_NONE)
            if add_header:
                writer.writerow(out)
            
            writer.writerow([out_statistic[key] for key in out])
    
    def print_params(self, params):
        line = "\n"
        index = 1
        path_info = []
        list_info = []
        for key in sorted(params):
            if params[key] and isinstance(params[key], str) and "/" in params[key]: 
                path_info.append(f'{key} : {params[key]}')
            elif params[key] and isinstance(params[key], list):
                list_info.append(f'{key} : {params[key]}')
            else:
                line = line + "{: <50} ".format(f'{key} : {params[key]}')
                if index % 3 ==0:
                    line = line + "\n"
                index+=1
        if len(list_info)>1:
            line = line + "\n"
            for list_item in list_info:
                line = line +  list_item + "\n"
        if len(path_info)>1:
            line = line + "\n"   
            for path in path_info:
                line = line +  path + "\n"
        return line 
    
    def update_stats(self, trails_stats, out_statistics):
        for key in out_statistics.keys():
            if not isinstance(out_statistics[key], list):
                if key not in trails_stats:
                    trails_stats[key] = []
                if isinstance(out_statistics[key], list) and utils.is_numeric(out_statistics[key]):
                    trails_stats[key].append(float(out_statistics[key]))
                elif utils.is_numeric(out_statistics[key]):
                    trails_stats[key].append(float(out_statistics[key]))
                else:
                    trails_stats[key].append(out_statistics[key])

    def executeExperiment(self, experiment_params):
        utils_processing.set_seed(0)

        params = experiment_params.copy()
        uniqueProcessingId = self.local_config_utils.getValue(config_const.CONF_SEC_EXPERIMENT, config_const.CONF_EXPERIMENT_UNIQUE_PROCESSING_ID)
        self.makeOutputFolder(uniqueProcessingId, params)
        
        trials_nbr =  self.local_config_utils.getValue(config_const.CONF_SEC_EXPERIMENT, config_const.CONF_EXPERIMENT_NBR_TRAILS, defaultValue=1)
        DF_RESULTS = pd.DataFrame(columns=["experiment_name","nbr_trials"])        
        logging.info(f'Experiment [{uniqueProcessingId}] params: {self.print_params(params)}')
        with_mlflow = params["mlflow.enable"]
        out_lost = None
        trials_stats = {}
        for trial in range(trials_nbr):
            utils_processing.set_seed(0+trial)
            learner = LearnerFactory.getLearner(params["learner.name"], params=params, mlflow=None)
            logging.info(f'Experiment trials [{trial}/{trials_nbr}]')
            params[config_const.CONF_EXPERIMENT_CURRENT_TRAIL_NUMBER] = trial
            if with_mlflow:
                mlflow.set_tracking_uri(MLFLOW_URL)
                mlflow.set_experiment(params["mlflow.experiment_name"])
                with mlflow.start_run() as run:
                    mlflow.set_tag("mlflow.runName", uniqueProcessingId)
                    out_lost, out_statistics  = learner.fit(tune=tune, trials_stats=trials_stats)
                    
            else:
                out_lost, out_statistics = learner.fit(tune=tune, trials_stats=trials_stats)
            torch.cuda.empty_cache()
            if trials_nbr>1:
                out_statistics["nbr_trials"]=trials_nbr
                self.update_stats(trials_stats, out_statistics)
            self.storeLearningResults(trial, out_statistics, experiment_params, uniqueProcessingId)
            DF_RESULTS = pd.concat([DF_RESULTS, pd.DataFrame(out_statistics["perf_per_epoch"])],ignore_index=True)
        self.storeLearningResults(trial, trials_stats, experiment_params, uniqueProcessingId, is_final_trial=True)

        DF_RESULTS.to_csv(os.path.join(self.getExperimentOutputPath(uniqueProcessingId, params),'out_'+time.strftime("%Y%m%d-%H%M%S")+'.csv'))          
        
        
        logging.info(f'Load model [{trials_stats["best_test_loss_per_trail_model"]}]')
        model = learner.loadModel(trials_stats["best_test_loss_per_trail_model"], "test_best_loss")
        tempalte_params = params.copy()
        predict_y, gold_y, emb_src, emb_trg = learner.do_predict(model)        
        emb_sim_eval = utils.embedding_similarity_evaluator(emb_src, emb_trg, gold_y)
        tempalte_params.update(emb_sim_eval)        
        
        df_evaluate = None
        if params[config_const.CONF_EXPERIMENT_VERIFY_ON_EVAL]:   
            df_evaluate = utils.evaluate_sts(param=params,model=model, learner=learner)
        
        alignments_results = self.evaluate_on_alignment(params, tempalte_params, trials_stats, learner)
        
        utils.save_report(predict_y, gold_y , df_evaluate, DF_RESULTS, params[config_const.CONF_MLFLOW_EXPERIMENT_NAME], experiment_params, params, tempalte_params, learner.data_holder.test_data_loader, alignments_results, True, self.getExperimentOutputPath(uniqueProcessingId, params))



    def evaluate_on_alignment(self, params, tempalte_params, trials_stats, learner):
        alignments_results = {}
        if params[config_const.CONF_EXPERIMENT_VERIFY_ON_ALIGNMENT]: 
            store_on_metrics =  params[config_const.CONF_MODEL_STORE_BEST_MODEL_ON_METRIC]       
            verify_on = [[trials_stats["best_test_loss_per_trail_model"], "test_best_loss","test_best_loss"]]
            for key in store_on_metrics:
                verify_on.append([trials_stats["best_test_"+key+"_per_trail_model"], "test_best_"+key,"test_best_"+key])
                
            copy_params = params.copy()
            alignment_datasets = params[config_const.CONF_ALIGNMENT_GRAPH_DATA_SET]
            if isinstance(alignment_datasets, str):
                alignment_datasets = [alignment_datasets]
            for index_dataset, alignment_dataset in enumerate(alignment_datasets):
                
                if config_const.CONF_DATASET_LIMIT in copy_params:
                    del copy_params[config_const.CONF_DATASET_LIMIT]
                    
                copy_params[config_const.CONF_DATASET_NAME] = alignment_dataset
                copy_params[config_const.CONF_DATASET_INIT_SINGLE_DATASET] ="test"
                data_holder = DataSetFactory.get_data_holder(copy_params)
                alignment_test_loader =  data_holder.test_data_loader 
                alignment_test_loader = DataLoader(alignment_test_loader.dataset, 
                        batch_size=params[config_const.CONF_DATASET_BATCH_SIZE], 
                        shuffle=False,
                        follow_batch=alignment_test_loader.follow_batch, 
                        exclude_keys=alignment_test_loader.exclude_keys
                        )   

                torch.cuda.empty_cache() 
                alignment_model = GNNModel(params=params)
                for index_verify, metric in enumerate(verify_on):
                    result_metric = str(index_verify)+"_"+str(index_dataset)+"_"+metric[2] + "_" + alignment_dataset
                    if metric[0]:
                        alignment_model.set_model(learner.loadModel(metric[0],metric[1]))
                    else:
                        if config_const.CONF_ALIGNMENT_GRAPH_MODEL_PATH in params and params[config_const.CONF_ALIGNMENT_GRAPH_MODEL_PATH]:
                            model =utils.load_model_from_path(learner.getModel(), params[config_const.CONF_ALIGNMENT_GRAPH_MODEL_PATH])
                            model.eval()
                            metric[0] = params[config_const.CONF_ALIGNMENT_GRAPH_MODEL_PATH].split("/")[-1]
                        alignment_model.set_model(model)
                    alignment_model.set_test_loader(alignment_test_loader)
                    alignment_model.set_learner(learner)
                    alignments_results[result_metric] = alignment_model.do_alignment()
                    top_metrics = {}           
                    for index_result, sim_metric in enumerate(alignments_results[result_metric]):

                        best_f_p, best_f_r, best_f_f1 = alignment_model.get_best_result(alignments_results[result_metric][sim_metric]["f_p"], alignments_results[result_metric][sim_metric]["f_r"], alignments_results[result_metric][sim_metric]["f_f1"])
                        
                        roc_auc = alignments_results[result_metric][sim_metric]["roc_auc"]
                        pr_auc = alignments_results[result_metric][sim_metric]["pr_auc"]
                                                        
                        best_alignments, golds = alignment_model.get_best_alignments(alignments_results[result_metric][sim_metric]["f_f1"], alignments_results[result_metric][sim_metric]["_range"], alignments_results[result_metric][sim_metric]["all_alignments"], alignments_results[result_metric][sim_metric]["all_gold"] )
                        alignments_results[result_metric][sim_metric]["best_alignment"] = best_alignments, golds
                        alignments_results[result_metric][sim_metric]["best_metrics"] = best_f_p, best_f_r, best_f_f1, roc_auc, pr_auc
                        top_metrics[result_metric+"%"+sim_metric] = best_f_f1
                        out = {}
                        out[sim_metric+"_"+result_metric+"_"+params[config_const.CONF_ALIGNMENT_MODEL]] = f'P: {best_f_p:.4f} R: {best_f_r:.4f} F1: {best_f_f1:.4f} ROC_AUC: {roc_auc:.4f} PR_AUC: {pr_auc:.4f}'
                        tempalte_params.update(out)
                        print(f'{params[config_const.CONF_ALIGNMENT_MODEL]}: Metric: {sim_metric}_{result_metric}[{metric[0][:30]}]  Prec {best_f_p:.4f} Rec {best_f_r:.4f} F1 {best_f_f1:.4f} ROC_AUC: {roc_auc:.4f} PR_AUC: {pr_auc:.4f}')
                        if sim_metric in 'prediction':
                            if "pred_sim_metric" not in tempalte_params:
                                tempalte_params["pred_sim_metric"] = str(f'{best_f_f1:.4f}')
                            tempalte_params["pred_sim_metric"]=tempalte_params["pred_sim_metric"]+"_"+str(f'{best_f_f1:.4f}')
                    top = sorted(top_metrics.items(), key=lambda x:x[1], reverse=True)[:3]
                    alignments_results[result_metric][top[0][0].split("%")[-1]]["the_best_results"] = True
                    if result_metric+"%prediction" not in top:
                        top.append((result_metric+"%prediction", top_metrics[result_metric+"%prediction"]))
                    top = list(zip(*top))[0]
                    for index_result, sim_metric in enumerate(list(alignments_results[result_metric])):
                        if result_metric+"%"+sim_metric not in top:
                            del  alignments_results[result_metric][sim_metric]
                    alignments_results[result_metric]["dataset_name"] = alignment_dataset         
        return alignments_results


class AbstractAlignmentModel(ABC):

    def __init__(self, params):
        self.params = params
        self.batch_size = 150
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.test_loader = None
        self.learner = None

    def set_model(self, model):
        self.model = model

    def set_test_loader(self, test_loader):
        self.test_loader = test_loader
    
    def set_learner(self, learner):
        self.learner = learner    
    
    @abstractmethod          
    def get_similarity_score(self):
        pass
    
    def get_data(self, reduce_to_aligned=True):
        path_to_ds = self.params[config_const.CONF_ALIGNMENT_DATA_SET]
        file = open(path_to_ds, 'r', encoding="utf-8")
        source = []
        target = []
        aligned = []
        is_eof = False
        while not is_eof:
            line = file.readline()
            if len(line) > 0:
                items = line.split("\t")
                aligned.append(items[0])
                source.append(items[3])
                target.append(items[4])
            else:
                is_eof = True 
        if reduce_to_aligned:
            for index in range(len(aligned)):
                if aligned[index] == 'partialAligned':
                    aligned[index] = 'aligned'
        return source, target, aligned
        
    def do_alignment(self):
        out = {}
        all_sim_scores_dic, all_gold  = self.get_similarity_score()
        for key in all_sim_scores_dic:
            logging.info(f'Evaluate alignment on metric [{key}]')
            f_p, f_r, f_f1, _range, mode_idx  = alig_func.evaluate_alignments_in_range(all_sim_scores_dic[key], all_gold, show_progress=True)
            out[key] = {}
            out[key]["f_p"] = f_p
            out[key]["f_r"] = f_r
            out[key]["f_f1"] = f_f1
            out[key]["_range"] = _range
            out[key]["mode_idx"] = mode_idx
            out[key]["all_alignments"] = all_sim_scores_dic[key]
            out[key]["all_gold"] = all_gold            
            out[key]["roc_auc"] = metrics.roc_auc_score(all_gold, all_sim_scores_dic[key])
            precision, recall, _ = metrics.precision_recall_curve(all_gold, all_sim_scores_dic[key])
            out[key]["pr_auc"] = metrics.auc(recall, precision)
        return out 
    
    def get_best_result(self, f_p, f_r, f_f1):
        mode_idx = np.argmax(f_f1)
        return f_p[mode_idx], f_r[mode_idx], f_f1[mode_idx]
    
    def get_best_alignments(self, f_f1, _range, all_sim_scores, all_gold):
        mode_idx = np.argmax(f_f1)
        alignments, golds =  alig_func.extract_alignments(all_sim_scores, all_gold, _range[mode_idx])
        return alignments, golds


class GNNModel(AbstractAlignmentModel):

    def __init__(self, params):
        super().__init__(params) 
        self.copy_params = params.copy()

        
    def get_similarity_score(self):
        out = {}
        predict_y, gold_y, emb_src, emb_trg = self.learner.do_predict(self.model, data_loader=self.test_loader)  
        if "prediction" in self.copy_params[config_const.CONF_ALIGNMENT_SIMILARITY_METRIC]:
            out["prediction"]=predict_y            
        if "cosine" in self.copy_params[config_const.CONF_ALIGNMENT_SIMILARITY_METRIC]:
            out["cosine"]=np.exp(- (paired_cosine_distances(emb_src, emb_trg)))
        if "manhattan" in self.copy_params[config_const.CONF_ALIGNMENT_SIMILARITY_METRIC]:  
            manhattan_distances = -paired_manhattan_distances(emb_src, emb_trg)
            out["manhattan"]=manhattan_distances
            out["manhattan_score"]=np.exp(manhattan_distances)
            out["manhattan_score_neg"]=np.exp(- manhattan_distances)        
        if "euclidean" in self.copy_params[config_const.CONF_ALIGNMENT_SIMILARITY_METRIC]:  
            euclidean_distances = -paired_euclidean_distances(emb_src, emb_trg)
            out["euclidean"]=euclidean_distances
            out["euclidean_score"]=np.exp(euclidean_distances)
            out["euclidean_score_neg"]=np.exp(- euclidean_distances)   
        if "dot" in self.copy_params[config_const.CONF_ALIGNMENT_SIMILARITY_METRIC]:         
            dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(emb_src, emb_trg)]  
            out["dot"]=dot_products
        return out, gold_y

    def get_model(params):
        return GNNModel(params)
        