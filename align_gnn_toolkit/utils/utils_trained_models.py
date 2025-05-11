#from torch_geometric.loader import DataLoader
#from learner import LearnerFactory
#from data_set.data_set_factory import DataSetFactory
#from engine import EngineFactory
#from utils import config_const
import os
#import torch_geometric
import torch
import json
from tqdm import tqdm
from math import sqrt
import scipy
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data_set.data_set_processor import DataSetProcessor
from matplotlib.backends.backend_pdf import PdfPages

import math

"""
def load_trained_model(model_type, time_stamp, params, data_holder, root):
    path = get_path_to_trained_model(model_type, time_stamp,root)
    model = LearnerFactory.getLearner(model_type, params=params).getModel()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
    
def get_model( model_type, time_stamp, root="."):
    path_to_models = os.path.join(root, "models")
    config_utils = EngineFactory().getConfigurationUtils(configurationFile=None, config_as_dict=get_configuration_as_dict(model_type, time_stamp,path_to_models))
    results = get_results_as_dict(model_type, time_stamp,path_to_models)
    engine = EngineFactory().getEngineType(config_utils)
    params = engine.getProcessingParameters()
    if not config_const.CONF_DATASET_PATH_TO_ROOT in params or not params[config_const.CONF_DATASET_PATH_TO_ROOT]:
        params[config_const.CONF_DATASET_PATH_TO_ROOT] = root
    data_holder = DataSetFactory.get_data_holder(params=params)
    model = load_trained_model(model_type, time_stamp, params,data_holder, path_to_models)
    test_data_set = data_holder.test_data_set
    test_data_set.graph_builder.initialize_text_processor()
    test_data_loader = DataLoader(test_data_set, 
                        batch_size=1, 
                        shuffle=False,
                        follow_batch=['x_s', 'x_t']
                        )
    return model, test_data_set, test_data_loader, results    
    
"""

def get_path_to_trained_model(model_type, time_stamp,root):
    return os.path.join(root, "Learner"+model_type, time_stamp, "model.pt")

def get_configuration_as_dict(model_type, time_stamp,root):
    with open(os.path.join(root, "Learner"+model_type, time_stamp, "experiment_params.json"), 'r') as j:
        return  json.loads(j.read())
    
def get_results_as_dict(model_type, time_stamp, root):
    with open(os.path.join(root, "Learner"+model_type, time_stamp, "results.json"), 'r') as j:
        return  json.loads(j.read())




"""
def get_predictions(model, train_data_loader):
    predict_y = []
    gold_y = []
    for index, batch in enumerate(tqdm(train_data_loader)):
        graph = batch
        gold_y.append((index, float(graph.y)))
        y_hat,_,_ =model(graph)
        predict_y.append((index, float(y_hat)))
    return predict_y, gold_y

def get_values_as_list(items, dim):
    return [x[dim] for x in items]
"""

def get_prediction_statistic(pred, gold):
    pearson_correlation = scipy.stats.pearsonr([x for x in pred], [x for x in gold])[0]
    spearman_correlation = scipy.stats.spearmanr([x for x in pred], [x for x in gold] )[0]
    #r2_scr = r2_score([x.detach().cpu() for x in pred], [x.cpu() for x in gold])
    r2_scr = r2_score([x for x in gold], [x for x in pred])
    rmse = sqrt(mean_squared_error([x for x in pred], [x for x in gold]))
    textstr = 'R2_Score=%.3f\nRMSE=%.3f\n$Pearson Correlation=%.3f$\n$Spearman Correlation=%.3f$' % (r2_scr, rmse, pearson_correlation, spearman_correlation)
    return textstr

def visualize_prediction(pred, gold, title=None):
    #pearson_correlation = scipy.stats.pearsonr([x[1] for x in gold_y], [x[1] for x in predict_y] )[0]
    #gold = get_values_as_list(gold_y, 1)
    #pred = get_values_as_list(predict_y, 1)
    
    #pearson_correlation = scipy.stats.pearsonr([x.detach().cpu() for x in pred], [x.cpu() for x in gold])[0]
    #spearman_correlation = scipy.stats.spearmanr([x.detach().cpu() for x in pred], [x.cpu() for x in gold] )[0]
    #r2_scr = r2_score([x.detach().cpu() for x in pred], [x.cpu() for x in gold])
    #rmse = sqrt(mean_squared_error([x.detach().cpu() for x in pred], [x.cpu() for x in gold]))
    #textstr = 'R2_Score=%.3f\nRMSE=%.3f\n$Pearson Correlation=%.3f$\n$Spearman Correlation=%.3f$' % (r2_scr, rmse, pearson_correlation, spearman_correlation)
    textstr = get_prediction_statistic(pred,gold)

    gold_y = [(i, x) for i,x in enumerate(gold)]
    predict_y = [(i, x) for i,x in enumerate(pred)]
    
    sorted_gold_y = sorted(gold_y, key=lambda x:x[1], reverse=False)
    sorted_pred_y  = [(i,predict_y[x[0]][1]) for i,x in enumerate(sorted_gold_y)]

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter([x[0] for x in sorted_pred_y], [x[1] for x in sorted_pred_y], color='DarkGreen', label='Predicted Similarity',
                alpha=0.5, edgecolors='none')
    ax.scatter([x[0] for x in sorted_pred_y], [x[1] for x in sorted_gold_y], color='DarkBlue', label='Similarity',
                alpha=0.2, edgecolors='none')
    ax.text(0.7, 0.1, textstr, fontsize=12,  ha='center', transform=ax.transAxes)
    ax.legend()
    if title:
        ax.set_title(title)
    return fig

def get_result_items(item):
    return [x for x in  item]

def get_residuals(predict_y, gold_y):
    #return (np.array([x[1] for x in gold_y]) -  np.array([x[1] for x in predict_y]))
    return np.subtract(get_result_items(predict_y), get_result_items(gold_y))
    
def visualize_residuals(predict_y, gold_y):
    fig, ax = plt.subplots(figsize=(10, 10))
    residuals = get_residuals(predict_y, gold_y)
    #sns.histplot(data=(np.array([x[1] for x in gold_y]) -  np.array([x[1] for x in predict_y])), ax=ax, kde=True, bins=10)
    #ax.axvline(x=np.mean((np.array([x[1] for x in gold_y]) -  np.array([x[1] for x in predict_y]))), color='r', ls='--', linewidth=2)
    sns.histplot(data=residuals, ax=ax, kde=True, bins=10)
    ax.axvline(x=np.mean(residuals), color='r', ls='--', linewidth=2)
    ax.legend()
    return fig

def insertChar(mystring, position, chartoinsert ):
    mystring   =  mystring[:position] + chartoinsert + mystring[position:] 
    return mystring  

def split_long_per_line(str, at_column):
    split_count = math.ceil(len(str)/at_column)
    for i in range(1,split_count):
        str = insertChar(str, at_column*i, "\n")
    return str

def get_data_by_index(data_loader, id):
    for index, data in enumerate(data_loader):
        if index == id:
            return data
    return None

def get_sentences(items, data_loader, predict_y,gold_y, as_list=False, is_shuffled=False):
    out = ""
    out_as_list = []
    data_set = data_loader.dataset
    
    if isinstance(data_loader.dataset, torch.utils.data.dataset.ConcatDataset):
        vocab = data_loader.dataset.datasets[0].data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab
    else:
        vocab = data_loader.dataset.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab
    for index in items:
        graph = None
        if is_shuffled:
            graph = get_data_by_index(data_loader, index)
        else: 
            graph = data_set[index]
        source, target = graph.get_sentences( vocab, ignore_unk=True, remove_roberta=True)
        if as_list:
            out_as_list.append([source, target])
        else:
            #out+=f'S: {source}\nT: {target}\nI: {index} G: {(data_set[index].y.item()):.3f} vs.P: { predict_y[index]:.3f} diff: {(data_set[index].y.item() - predict_y[index]):.3f}\n\n'
            out+=f'S: {split_long_per_line(source,110)}\nT: {split_long_per_line(target,110)}\nI: {index} G: {(gold_y[index]):.3f} vs.P: { predict_y[index]:.3f} diff: {(gold_y[index] - predict_y[index]):.3f}\n\n'
    if as_list:
        return out_as_list
    else:
        return out
MAX_RETURN=50
def get_residuals_in_range(min, max, k_items, predict_y, gold_y):
    #residual=np.array([x[1] for x in gold_y]) -  np.array([x[1] for x in predict_y])
    residual = np.array(get_residuals(predict_y, gold_y))
    indices = np.logical_and(residual <= max, residual >=min)
    if k_items:
        return np.where(indices)[0][:k_items]
    else:
        return np.where(indices)[0][:MAX_RETURN]


def get_predicted_in_range(min, max, k_items, predict_y):
    pre = np.array(get_result_items(predict_y))
    indices = np.logical_and(pre <= max, pre >=min)
    if k_items:
        return np.where(indices)[0][:k_items]
    else:
        return np.where(indices)[0][:MAX_RETURN]

def get_overheat_predicted(threshold, k_items, predict_y, gold_y):
    pre = np.array(get_result_items(predict_y))
    gold = np.array(get_result_items(gold_y))
    indices = np.logical_and(pre >= gold, pre - gold>=threshold)
    if k_items:
        return np.where(indices)[0][:k_items]
    else:
        return np.where(indices)[0][:MAX_RETURN]

def get_overcold_predicted(threshold, k_items, predict_y, gold_y):
    pre = np.array(get_result_items(predict_y))
    gold = np.array(get_result_items(gold_y))
    indices = np.logical_and(pre <= gold, gold - pre  >=threshold)
    if k_items:
        return np.where(indices)[0][:k_items]
    else:
        return np.where(indices)[0][:MAX_RETURN]




def get_dict_content(dict_to_display, nbr_tabs=1):
    out = ""
    count =0
    for index, key in enumerate(dict_to_display.keys()):
        if dict_to_display[key]:
            out+=f'{key} : {dict_to_display[key]}'
            if count % nbr_tabs ==0:
                out+="\n"
            else:
                out+="\t\t\t\t"
            count+=1
    return out
    

def get_experiment_info(data_set):
    return get_dict_content(data_set.params)

def get_experiment_result(results):
    return get_dict_content(results)

def get_text_page(content,subject,size):
    page = plt.figure(figsize=size)
    page.clf()
    #page.text(0.1,0.1,content, transform=page.transFigure, size=11, ha="left")
    page.text(0.05, 0.95, subject, fontsize=12,  ha='left', va='top', transform=page.transFigure)
    page.text(0.05, 0.90, content, fontsize=11,  ha='left', va='top', transform=page.transFigure)
    return page
    


def generate_report(model_type, time_stamp,root,predict_y, gold_y, data_set,results):
    with PdfPages(os.path.join(root,"models", "Learner"+model_type, time_stamp, f'report_{model_type}_{time_stamp}.pt')) as pdf:
        pdf.savefig(get_text_page(get_experiment_info(data_set), "Experiment settings:", (10,10)))
        plt.close()
        pdf.savefig(get_text_page(get_experiment_result(results), "Experiment results:", (10,10)))
        plt.close()
        pdf.savefig(visualize_prediction(predict_y, gold_y))
        plt.close()
        pdf.savefig(visualize_residuals(predict_y, gold_y))
        plt.close()
        residuals = get_residuals(predict_y,gold_y)
        bins = np.histogram_bin_edges(residuals, bins=10)
        prev = -1
        current =0
        for index, bin in enumerate(bins):
            current=bin
            #out = f'Residual between {prev:.2f} and {current:.2f}'
            out = get_sentences(get_residuals_in_range(prev,current, 5, predict_y, gold_y), data_set, predict_y, gold_y)
            pdf.savefig(get_text_page(out, f'Residual between {prev:.2f} and {current:.2f} \n', (10,10)))
            plt.close()
            prev=current
        out = get_sentences(get_residuals_in_range(prev,1, 5, predict_y, gold_y), data_set, predict_y, gold_y)
        pdf.savefig(get_text_page(out,f'Residual between {current:.2f} and {1}', (10,10)))
        plt.close()
        pdf.savefig(get_text_page(get_sentences(get_overheat_predicted(0.4, 5, predict_y, gold_y), data_set, predict_y, gold_y), "Overheat evidences:", (10,10)))
        plt.close()
        pdf.savefig(get_text_page(get_sentences(get_overcold_predicted(0.4, 5, predict_y, gold_y), data_set, predict_y, gold_y), "Overcold evidences:", (10,10)))
        plt.close()
