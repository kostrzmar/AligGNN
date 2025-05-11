from scipy.stats import pearsonr 
from scipy.stats import spearmanr 
import numpy as np
import os
import json
from datetime import datetime
import torch
from utils import config_const
from data_set.data_set_factory import DataSetFactory
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import random
import time
import pandas as pd
from utils import utils_trained_models
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
from utils import alignment_func as alig_func
from sklearn import metrics
import torch_geometric
import torch
import spacy
import psutil
import logging
import numpy as np
import random
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from scipy.ndimage.filters import gaussian_filter1d



def get_mean_from_stats(dict_list):
    mean_dict = {}
    for key in ["R2_score", "pearsonr", "SpearmanR", "MQE", "MAE"]:
        val = [d[key] for d in dict_list]
        type = dict_list[0]["type"]
        mean_dict[type+"_"+key+"_mean"] = np.mean(val, axis=0).item()
        mean_dict[type+"_"+key+"_std"] =np.std(val, axis=0).item()
        mean_dict[type+"_"+key+"_min"] =np.min(val, axis=0).item()
        mean_dict[type+"_"+key+"_max"] =np.max(val, axis=0).item()
    return mean_dict



def get_experiment_path(experiment_name, time_stamp=None, experiment_timestamp=""):
    if time_stamp:
        path = os.path.join("./models/",experiment_name,experiment_timestamp, time_stamp)
    else:
        path = os.path.join("./models/",experiment_name,experiment_timestamp, datetime.now().strftime("%m%d_%H%M"))
    if not os.path.exists(path):
        os.makedirs(path)
    return path



def save_experiment_params(experiment_path, params):
    os.makedirs(experiment_path, exist_ok=True)  
    dic = json.dumps(params)
    f = open(os.path.join(experiment_path, "experiment_params.json"),"w")
    f.write(dic)
    f.close()

def store_per_per_epoch(perf_per_epoch, test_error, val_error, epoch, lr,loss, val_loss,test_loss, experiment_name, model_metrics, calculate_mean=True):
    inter = {}
    inter["test_error"] = test_error
    inter["val_error"] = val_error
    inter["epoch"] = epoch
    inter["nbr_trials"] = model_metrics["nbr_trials"]
    inter["experiment_name"] = experiment_name
    inter["lr"] = lr
    inter["loss"] = loss 
    inter["val_loss"] = val_loss 
    inter["test_loss"] = test_loss 
    
    def change_dict_key(d, old_key, new_key, default_value=None):
        d[new_key] = d.pop(old_key, default_value)
    
    
    for items in model_metrics["per_epoch"][epoch]:
        if calculate_mean:
            inter.update(get_mean_from_stats(items))
        else:
            new_items = {}
            type = items["type"]
            new_items["type"] = type
            for key in items:
                if key not in "type":
                    new_key = type+"_"+key
                    new_items[new_key] = items[key]
            inter.update(new_items)
    perf_per_epoch.append(inter)

def save_json(path, file_name, values):
    if values:
        to_save  = values.copy()
        for key in to_save:
            to_save[key] = str(to_save[key])
        with open(os.path.join(path, file_name), "w") as wf:
            json.dump(to_save, wf, indent=4)  
    
    
def save_model_raw(model,experiment_path, stats, type=""): 
    save_json(experiment_path, "results_"+type+".json", stats)
    torch.save(model.state_dict(), os.path.join(experiment_path, "model_"+type+".pt"))    
    
    
def save_model(model,experiment_path, stats, type=""): 
    with open(os.path.join(experiment_path, "results"+type+".json"), "w") as wf:
        json.dump(get_mean_from_stats(stats), wf, indent=4)  
    torch.save(model.state_dict(), os.path.join(experiment_path, "model"+type+".pt"))
    

def load_model(model, experiment_path, type=""):
    model = load_model_from_path(model, os.path.join(experiment_path, "model_"+type+".pt"))
    model.eval()
    return model
    
def load_model_from_path(model, path):
    model.load_state_dict(torch.load(path))
    return model

        


def get_model_params(params):
        return  {k: v for k, v in params.items() if k.startswith("model.")}

def get_feature_size(data_holder):
    if data_holder.train_data_loader:
        return data_holder.train_data_set[0].x_s.shape[1]
    else:
        return data_holder.test_data_set[0].x_s.shape[1]

def get_edge_feature_size(data_holder):
    if data_holder.train_data_loader:
        return data_holder.train_data_set[0].edge_attr_s.shape[1] 
    else:  
        return data_holder.test_data_set[0].edge_attr_s.shape[1]

def show_direction(loss, best_loss):
    l_ch = ""
    if best_loss is None or loss < best_loss:
        l_ch = "↓"
    elif loss > best_loss:
        l_ch = "↑"
    else:
        l_ch = "↔"
    return l_ch

def show_process_info(model):
    print("\nModel architecture:")
    print(model)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'Total parameters: {total_param}')
    
    
def convert_results(model_metrics):
    out = {}
    for key in model_metrics:
        if key not in ("perf_per_epoch"):
            if key in ("best_val_error","test_error","train_time", "experiment", "nbr_trials"):
                out[key] = model_metrics[key]
            else:
                out[key] = f'{np.average(model_metrics[key]):.3f}+/-{np.std(model_metrics[key]):.3f} [{np.min(model_metrics[key]):.3f}][{np.max(model_metrics[key]):.3f}]'
    return out

def get_final_performance_raw(dict_list, types):
    mean_dict = {}
    for key in types.keys():
        _val = [d[key] for d in dict_list if key in d]
        val = _val[min(3,len(_val))*(-1):]
        _type = types[key]
        for type in _type:
            if type == "mean":
                mean_dict[key+"_"+type] = np.mean(val, axis=0).item()
            elif type =="std":
                mean_dict[key+"_"+type] =np.std(val, axis=0).item()
            elif type =="min":
                mean_dict[key+"_"+type] =np.min(_val, axis=0).item()
            elif type =="max":
                mean_dict[key+"_"+type] =np.max(_val, axis=0).item()
    return mean_dict    

def get_final_performance(dict_list, types):
    mean_dict = {}
    for key in types.keys():
        val = [d[key] for d in dict_list if key in d]
        val = val[min(3,len(val))*(-1):]
        type = types[key]
        if type == "mean":
            mean_dict[key+"_"+type] = np.mean(val, axis=0).item()
        elif type =="std":
            mean_dict[key+"_"+type] =np.std(val, axis=0).item()
        elif type =="min":
            mean_dict[key+"_"+type] =np.min(val, axis=0).item()
        elif type =="max":
            mean_dict[key+"_"+type] =np.max(val, axis=0).item()
    return mean_dict


def show_figure(columns, titles, RESULTS):
    plt.figure(figsize=(10*len(columns),10))
    plt.rcParams["figure.figsize"] = [10.00*len(columns), 8]
    plt.rcParams["figure.autolayout"] = True
    if len(columns)==1:
        fig, ax = plt.subplots(1)
        ax.title.set_text(titles[0])
        sns.set_style("ticks",{'axes.grid' : True})
        sns.lineplot( x="epoch", y=columns[0], data=RESULTS, markers=False, dashes=False, hue="nbr_trials", palette="rocket", legend="full")
    else:
        fig, ax = plt.subplots(1, 2)
        ax[0].title.set_text(titles[0])
        ax[1].title.set_text(titles[1])
        sns.set_style("ticks",{'axes.grid' : True})
        sns.lineplot( x="epoch", y=columns[0], data=RESULTS, markers=False, dashes=False, hue="nbr_trials", palette="rocket", legend="full", ax=ax[0])
        sns.lineplot( x="epoch", y=columns[1], data=RESULTS, markers=False, dashes=False, hue="nbr_trials", palette="mako", legend="full", ax=ax[1])
    return fig    


def show_alignment_diagram(metric_name, f_p, f_r, f_f1, _range ):
    fig, ax = plt.subplots(figsize=(10,4))
    sns.lineplot(y=gaussian_filter1d(f_p, sigma=1), x=_range, ax=ax, label="Precision", linestyle="-")
    sns.lineplot(y=gaussian_filter1d(f_r, sigma=1), x=_range, ax=ax, label="Recall", linestyle="-")
    sns.lineplot(y=gaussian_filter1d(f_f1, sigma=0.4), x=_range, ax=ax, label="F1", linestyle="-")
    mode_idx = np.argmax(f_f1)
    ax.vlines(_range[mode_idx], 0, f_f1[mode_idx], ls='--', color="red")
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'[{metric_name}] F1: {f_f1[mode_idx]:.2f} [Threshold: {_range[mode_idx]:.2f}]', loc='left', color='blue', size=14)
    return fig

def plot_roc_curve(fpr, tpr, precision, recall, auc, pr_auc):
    fig, ax = plt.subplots(figsize=(10,4))
    plt.plot(fpr, tpr, color='orange', label=f'ROC, AUC={auc:.2f}' )
    plt.plot(precision, recall, color='green', label=f'PRC, PR_AUC={pr_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate / Recall')
    plt.ylabel('True Positive Rate / Precision')
    plt.title(f'ROC & PRC Curves [AUC: {auc:.2f} / PR_AUC: {pr_auc:.2f}]', loc='left', color='blue', size=14)
    plt.legend(loc=0)
    return fig



def get_text_page(content,subject,size):
    page = plt.figure(figsize=size)
    page.clf()
    page.text(0.05, 0.95, subject, fontsize=12,  ha='left', va='top', transform=page.transFigure)
    page.text(0.05, 0.90, content, fontsize=11,  ha='left', va='top', transform=page.transFigure)
    return page

def get_table(df_context,subject, size):
    page = plt.figure(figsize=size)
    ax =  page.gca()
    ax.axis('off')
    ax.set_title(subject)
    tab2 = ax.table(cellText=df_context.values,rowLabels=df_context.index, colLabels=df_context.columns, loc='center', cellLoc='center')
    tab2.auto_set_column_width(col=list(range(len(df_context.columns)))) 
    return page

def get_experiment_info(experiment, tempalte_params):
    out = ""
    dics = get_params(experiment,tempalte_params)
    for key in dics:
        out+=f'{key} - {dics[key]}\n'
    return out  

def get_experiment_info_per_page(experiment, tempalte_params, line_per_page):
    out_array = []
    dics = get_params(experiment,tempalte_params)
    key_sorted = sorted(dics)
    for key in key_sorted:
        out_array.append(f'{key} - {dics[key]}\n')
    
    text_per_page = [out_array[i:i + line_per_page] for i in range(0, len(out_array), line_per_page)]
    
    out_text = []
    for page in text_per_page:
        out  = ""
        for item in page:
            out+=item
        out_text.append(out)
    return out_text  

def get_params(experiment, tempalte_params):
    params = tempalte_params.copy()
    for key in experiment:
        params[key] = experiment[key]
    return params


def display_metric(model_metrics):
    for key in model_metrics:
        items = np.array(model_metrics[key])
        print(f'Key {key} avg: {np.average(items):.3f} sdv: {np.std(items):.3f} max: {np.max(items):.3f}')

def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    
def evaluate_sts(param, model, learner):
    print("Evaluating...")
    results = {}
    similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))
    eval_param = param.copy()
    default_eval = ['sts12',"sts13","sts14","sts15","sts16", "stsb", "sick"]
    if config_const.CONF_EXPERIMENT_VERIFY_ON_EVAL_DS in param and param[config_const.CONF_EXPERIMENT_VERIFY_ON_EVAL_DS]:
        default_eval  = param[config_const.CONF_EXPERIMENT_VERIFY_ON_EVAL_DS]
    for eval in default_eval:
        torch.cuda.empty_cache() 
        eval_param[config_const.CONF_DATASET_NAME] = eval
        eval_param[config_const.CONF_DATASET_INIT_SINGLE_DATASET] = "test"
        data_holder = DataSetFactory.get_data_holder(eval_param)
        test_data_loader = data_holder.test_data_loader
        
        all_sys_scores = []
        all_pred_score  = []
        all_gs_scores = []

        device = "cpu"
        model.to(device)
        y_pred, gold_y, emb_s, emb_y = learner.do_predict(model, data_loader=test_data_loader)
        for kk in range(emb_s.shape[0]):
            all_sys_scores.append(similarity(emb_s[kk], emb_y[kk]))
            all_pred_score.append(y_pred[kk])  
            all_gs_scores.append(gold_y[kk])         
        """
        for data in test_data_loader:
            model.eval()
            data = data.to(device)
            
            #y_pred, emb_s, emb_y, _, _ = model(data)
            for kk in range(emb_s.shape[0]):
                sys_score = similarity(emb_s[kk].cpu().detach().numpy(), emb_y[kk].cpu().detach().numpy())
                all_sys_scores.append(sys_score)
                all_pred_score.append(y_pred[kk].cpu().item())  
                all_gs_scores.append(data[kk].y.cpu().item()) 
        """
        per = pearsonr(all_sys_scores, all_gs_scores)
        spr =  spearmanr(all_sys_scores, all_gs_scores)
        per_pred = pearsonr(all_pred_score, all_gs_scores)
        spr_pred =  spearmanr(all_pred_score, all_gs_scores)

        results[eval_param[config_const.CONF_DATASET_NAME]] = {
            'pearson': f'{per[0]:.4f}' , 
            'pearson p_val': f'{per[1]:.4f}' , 
            'pearson_pred': f'{per_pred[0]:.4f}' , 
            'pearson_pred p_val': f'{per_pred[1]:.4f}' , 
            'spearman': f'{spr[0]:.4f}', 
            'spearman p_val': f'{spr[1]:.4f}', 
            'spearman_pred': f'{spr_pred[0]:.4f}', 
            'spearman_pred p_val': f'{spr_pred[1]:.4f}',
            'nsamples': len(all_sys_scores)}
        print('%s : pearson = %s (pred: %s), spearman = %s (pred: %s)' %
                            (eval_param[config_const.CONF_DATASET_NAME], 
                            results[eval_param[config_const.CONF_DATASET_NAME]]['pearson'],
                            results[eval_param[config_const.CONF_DATASET_NAME]]['pearson_pred'],
                            results[eval_param[config_const.CONF_DATASET_NAME]]['spearman'],
                            results[eval_param[config_const.CONF_DATASET_NAME]]['spearman_pred']
                            ))
    return pd.DataFrame(results)


def save_report(predict_y, gold_y , df_evaluate, DF_RESULTS, experiment_name, experiment, params, tempalte_params, test_loader, alignments_results=None, do_training=True, output_path="./experiment"):
    lr_exp = [] 
    loss_exp = []
    val_loss_exp = []
    test_loss_exp = []
    error_exp = []
    mae_exp = []
    mqe_exp = []
    pear_exp = []
    spear_exp = []
    r2_exp =  []
    if do_training:
        if isinstance(test_loader.dataset, torch.utils.data.dataset.ConcatDataset):
            for dataset in test_loader.dataset.datasets:
                dataset.graph_builder.initialize_text_processor()
                
        else:    
            test_loader.dataset.graph_builder.initialize_text_processor()
    #for experiment in experiments:
    RESULTS = DF_RESULTS.loc[DF_RESULTS['experiment_name'] == experiment_name]

    if do_training:
        lr_exp.append(show_figure(["lr"], ["Learning Rate"], RESULTS))
        loss_exp.append(show_figure(["loss"], ["Train Loss"], RESULTS))
        val_loss_exp.append(show_figure(["val_loss"], ["Verification Loss"], RESULTS))
        test_loss_exp.append(show_figure(["test_loss"], ["Test Loss"], RESULTS))
        
        error_exp.append(show_figure(["test_error","val_error"], ["Test error", "Val error"],RESULTS))
        mae_exp.append(show_figure(["test_MAE" ,"val_MAE" ], ["Test MAE" , "Val MAE"],RESULTS))
        mqe_exp.append(show_figure(["test_MQE" ,"val_MQE" ], ["Test MQA" , "Val MQA"],RESULTS))
        pear_exp.append(show_figure(["test_pearsonr" ,"val_pearsonr" ], ["Test Pearsonr" , "Val Pearsonr"],RESULTS))
        spear_exp.append(show_figure(["test_SpearmanR" ,"val_SpearmanR" ], ["Test SpearmanR" , "Val SpearmanR" ],RESULTS))
        r2_exp.append(show_figure(["test_R2_score" ,"val_R2_score" ], ["Test R2_score" , "Val R2_score"],RESULTS))
        
        prediction_fig = utils_trained_models.visualize_prediction(predict_y, gold_y)
        residuals_fig = utils_trained_models.visualize_residuals(predict_y, gold_y)
        tempalte_params["Best results"] = utils_trained_models.get_prediction_statistic(predict_y, gold_y)

    sim_metric = ""
    if "pred_sim_metric" in tempalte_params:
        sim_metric = "_"+tempalte_params["pred_sim_metric"]
    experiment_report_name = experiment_name+sim_metric+'_'+ time.strftime("%H%M%S")+'.pdf'
    tempalte_params["Experiment Name"] = experiment_report_name
    predictions_fig = []
    for experiment_name in alignments_results.keys():
        predictions_fig.append(utils_trained_models.visualize_prediction(alignments_results[experiment_name]["prediction"]['all_alignments'], alignments_results[experiment_name]["prediction"]['all_gold'], experiment_name+" prediction"))
        if "cosine" in alignments_results[experiment_name]:
            predictions_fig.append(utils_trained_models.visualize_prediction(alignments_results[experiment_name]["cosine"]['all_alignments'], alignments_results[experiment_name]["cosine"]['all_gold'], experiment_name+" cosine"))
        
    with PdfPages(os.path.join(output_path, 'report_'+experiment_report_name)) as pdf:
        
        if do_training:
            for i in range(len(lr_exp)):
                text_per_page = get_experiment_info_per_page(experiment, tempalte_params=tempalte_params, line_per_page=40)
                for text in text_per_page:                
                    pdf.savefig(get_text_page(text, "Experiment settings:", (10,10)))
                    plt.close()
            
                pdf.savefig(lr_exp[i])    
                plt.close()
                pdf.savefig(loss_exp[i])
                plt.close()
                pdf.savefig(val_loss_exp[i])
                plt.close()
                pdf.savefig(test_loss_exp[i])

                plt.close()
                pdf.savefig(error_exp[i])
                plt.close()
                pdf.savefig(mae_exp[i])
                plt.close()
                pdf.savefig(mqe_exp[i])
                plt.close()
                pdf.savefig(pear_exp[i])
                plt.close() 
                pdf.savefig(spear_exp[i])
                plt.close()  
                pdf.savefig(r2_exp[i])
                plt.close()

            pdf.savefig(prediction_fig)
            plt.close() 
            pdf.savefig(residuals_fig)
            plt.close()

        else:
            text_per_page = get_experiment_info_per_page(experiment, tempalte_params=tempalte_params, line_per_page=40)
            for text in text_per_page:                
                pdf.savefig(get_text_page(text, "Experiment settings:", (10,10)))
                plt.close()



        for fig in predictions_fig:
            pdf.savefig(fig)
            plt.close() 

        if do_training:
            residuals = utils_trained_models.get_residuals(predict_y,gold_y)
            bins = np.histogram_bin_edges(residuals, bins=10)
            prev = -1
            current =0
            for index, bin in enumerate(bins):
                current=bin
                out = utils_trained_models.get_sentences(utils_trained_models.get_residuals_in_range(prev,current, 5, predict_y, gold_y), test_loader, predict_y, gold_y)
                
                pdf.savefig(get_text_page(out, f'Residual between {prev:.2f} and {current:.2f} \n', (10,10)))
                plt.close()
                show_details=False
                if show_details:
                    all_out = utils_trained_models.get_sentences(utils_trained_models.get_residuals_in_range(prev,current, None, predict_y, gold_y), test_loader, predict_y, gold_y, as_list=True)
                    if len(all_out)>0:
                        ds = Dataset("Test")
                        source, target = [], []
                        for item in all_out:
                            source.append(item[0]) 
                            target.append(item[1])
                        pdf.savefig(get_text_page(ds.getStatisticAsString(source, target), f'Stats of Residual between {prev:.2f} and {current:.2f} \n', (10,10)))
                        plt.close()
                prev=current 
            out = utils_trained_models.get_sentences(utils_trained_models.get_residuals_in_range(prev,1, 5, predict_y, gold_y), test_loader, predict_y, gold_y)
            if len(out) > 1: 
                pdf.savefig(get_text_page(out,f'Residual between {current:.2f} and {1}', (10,10)))
                plt.close()
            pdf.savefig(get_text_page(utils_trained_models.get_sentences(utils_trained_models.get_overheat_predicted(0.4, 5, predict_y, gold_y), test_loader, predict_y, gold_y), "Overheat evidences:", (10,10)))
            plt.close()
            pdf.savefig(get_text_page(utils_trained_models.get_sentences(utils_trained_models.get_overcold_predicted(0.4, 5, predict_y, gold_y), test_loader, predict_y, gold_y), "Overcold evidences:", (10,10)))
            plt.close()

        if df_evaluate is not None:
            pdf.savefig(get_table(df_evaluate,"Embedding Similarity", (10,10)))
            plt.close()    
        if alignments_results is not None:
            keys = sorted(alignments_results.keys())
            loc_dataset_name = None
            alig_statistic = []
            for index in range(len(keys)):
            #for key in alignments_results:
                key = keys[index]
                copy_params = params.copy()
                if not loc_dataset_name or loc_dataset_name != alignments_results[key]["dataset_name"]:
                    
                    loc_dataset_name = alignments_results[key]["dataset_name"]
                    print(f'Reading {loc_dataset_name} dataset')
                    copy_params[config_const.CONF_DATASET_NAME] = loc_dataset_name
                    if config_const.CONF_DATASET_LIMIT in params:
                        del copy_params[config_const.CONF_DATASET_LIMIT]
                    data_holder = DataSetFactory.get_data_holder(copy_params)
                    alignment_test_loader =  data_holder.test_data_loader                  
                del alignments_results[key]["dataset_name"]
                for metric in alignments_results[key]:
                    alignment_stats_row = {}
                    pdf.savefig(show_alignment_diagram(key+"_"+metric, alignments_results[key][metric]["f_p"], alignments_results[key][metric]["f_r"], alignments_results[key][metric]["f_f1"], alignments_results[key][metric]["_range"] ))
                    plt.close()  
                    all_alignments = alignments_results[key][metric]["all_alignments"]
                    all_gold = alignments_results[key][metric]["all_gold"]
                    _range = alignments_results[key][metric]["_range"]
                    mode_idx = alignments_results[key][metric]["mode_idx"]
                    alignments, golds = alignments_results[key][metric]["best_alignment"]
                    best_f_p, best_f_r, best_f_f1, roc_auc, pr_auc = alignments_results[key][metric]["best_metrics"]  
                    predictions = set([x[0] for x in alignments])
                    actuals = set([x[0] for x in golds])
                    TP = predictions.intersection(actuals)
                    FN = actuals.difference(predictions)
                    FP = predictions.difference(actuals)

                    
                    fpr, tpr, thresholds = metrics.roc_curve(all_gold, all_alignments)
                    precision, recall, thresholds = metrics.precision_recall_curve(all_gold, all_alignments)                
                    pdf.savefig(plot_roc_curve(fpr, tpr, precision, recall, roc_auc, pr_auc))
                    plt.close()
                      
                    page_content = f'TP:{len(TP)}, FP:{len(FP)}, FN:{len(FN)}, PRED_COUNT:{len(predictions)}, GOLD_COUNT:{len(actuals)}\n' 
                    page_content += "\n"
                    page_content += f'P:{best_f_p:.2f}, R:{best_f_r:.2f}, F1:{best_f_f1:.2f}, ROC_AUC:{roc_auc:.2f} PR_AUC:{pr_auc:.2f} \n'
                    alignment_stats_row['KEY'] = key
                    alignment_stats_row['METRIC'] = metric
                    alignment_stats_row['TP'] = len(TP)
                    alignment_stats_row['FP'] = len(FP)
                    alignment_stats_row['FN'] = len(FN)
                    alignment_stats_row['PRED_COUNT'] = len(predictions)
                    alignment_stats_row['GOLD_COUNT'] = len(actuals)
                    alignment_stats_row['P'] = best_f_p
                    alignment_stats_row['R'] = best_f_r
                    alignment_stats_row['F1'] = best_f_f1
                    alignment_stats_row['ROC_AUC'] = roc_auc
                    alignment_stats_row['PR_AUC'] = pr_auc
                    alig_statistic.append(alignment_stats_row)
                    
                    pdf.savefig(alig_func.get_text_page(page_content, f'', (10,2)))
                    plt.close()    
                    
                    pdf.savefig(alig_func.get_histogram([x.item() for x in alignments_results[key][metric]["all_alignments"]]))
                    plt.close()
                    
                    if "the_best_results" in  alignments_results[key][metric]:
                                            
                        alig_func.get_report_for_evidences(pdf, FN, "FN", all_alignments, all_gold, _range, mode_idx,  bins=10, max_examples=7, new_line_column=110, test_loader=alignment_test_loader)
                        
                        alig_func.get_report_for_evidences(pdf, FP, "FP", all_alignments, all_gold, _range, mode_idx,  bins=10, max_examples=7, new_line_column=110, test_loader=alignment_test_loader)
            pd.DataFrame.from_dict(alig_statistic).to_csv(os.path.join(output_path,'alignments_results_out_'+time.strftime("%Y%m%d-%H%M%S")+'.csv'))


def embedding_similarity_evaluator(embeddings1, embeddings2, gold_y, log_results = True):
    cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
    manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
    euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
    dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]


    eval_pearson_cosine, _ = pearsonr(gold_y, cosine_scores)
    eval_spearman_cosine, _ = spearmanr(gold_y, cosine_scores)

    eval_pearson_manhattan, _ = pearsonr(gold_y, manhattan_distances)
    eval_spearman_manhattan, _ = spearmanr(gold_y, manhattan_distances)

    eval_pearson_euclidean, _ = pearsonr(gold_y, euclidean_distances)
    eval_spearman_euclidean, _ = spearmanr(gold_y, euclidean_distances)

    eval_pearson_dot, _ = pearsonr(gold_y, dot_products)
    eval_spearman_dot, _ = spearmanr(gold_y, dot_products)
    if log_results:
        print("Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_cosine, eval_spearman_cosine))
        print("Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_manhattan, eval_spearman_manhattan))
        print("Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_euclidean, eval_spearman_euclidean))
        print("Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_dot, eval_spearman_dot))
    out = {}
    out["Embeddings Cosine-Similarity"] = f'Pearson: {eval_pearson_cosine:.4f} Spearman:{eval_spearman_cosine:.4f}'
    out["Embeddings Manhattan-Distance"] = f'Pearson: {eval_pearson_manhattan:.4f} Spearman:{eval_spearman_manhattan:.4f}'
    out["Embeddings Euclidean-Distance"] = f'Pearson: {eval_pearson_euclidean:.4f} Spearman:{eval_spearman_euclidean:.4f}'
    out["Embeddings Dot-Product-Similarity"] = f'Pearson: {eval_pearson_dot:.4f} Spearman:{eval_spearman_dot:.4f}'
    return out


class ModelPerformance:
    def __init__(self):
        self.stats = {} 
        self.init()
    
    def init(self):
        self.set("val", "loss", None)
        self.set("val", "best_loss", None)
        self.set("test", "loss", None)
        self.set("test", "best_loss", None)    
        
    def get(self, type, key):
        if type+"_"+key in self.stats:
            return self.stats[type+"_"+key]
        return None

    def set(self, type, key, value):
        self.stats[type+"_"+key] = value
        
        
from tqdm import tqdm
import torch.nn.functional as F 
def pad_dataset(dataset):
    max_label_t = -1
    max_label_s = -1
    for item in tqdm(dataset):
        if item.node_labels_t.shape[1] > max_label_t:
            max_label_t = item.node_labels_t.shape[1]
        if item.node_labels_s.shape[1] > max_label_s:
            max_label_s = item.node_labels_s.shape[1]
    print(f'max pad for source: {max_label_s} target: {max_label_t}')
            
    new_dataset = []
    for data in tqdm(dataset):
        org = data._store['node_labels_t']
        data._store['node_labels_t'] = F.pad(org, (1,max_label_t-org.shape[1]), 'constant', -1)
        org = data._store['node_labels_s']
        data._store['node_labels_s'] = F.pad(org, (1,max_label_t-org.shape[1]), 'constant', -1)
        new_dataset.append(data)
    return new_dataset


def print_params(params):
    line = "\n"
    index = 1
    for key in sorted(params):
        line = line + "{: <50} ".format(f'{key} : [{params[key]}]')
        if index % 3 ==0:
            line = line + "\n"
        index+=1
    print(line) 

def generate_unique_exp_name(params):
    # code for folder_name_generator
    configs = [
    config_const.CONF_TRAIN_EPOCH_NUMBER,
    config_const.CONF_LEARNER_NAME,
    config_const.CONF_DATASET_BATCH_SIZE,
    config_const.CONF_OPTIMIZER_LEARNING_RATE,
    config_const.CONF_OPTIMIZER_LEARNING_WARMUP_STEPS,
    config_const.CONF_OPTIMIZER_LEARNING_WARMUP_INIT_LR,
    config_const.CONF_MODEL_DROPOUT_RATE,
    config_const.CONF_MODEL_SIMGNN_GNN_OPERATOR,
    config_const.CONF_MODEL_SIMGNN_FILTERS_1 ,
    config_const.CONF_MODEL_SIMGNN_FILTERS_2,
    config_const.CONF_MODEL_SIMGNN_FILTERS_3,
    config_const.CONF_GRAPH_BUILDER_BIDIRECTED,
    config_const.CONF_GRAPH_BUILDER_SELFLOOP,
    config_const.CONF_GRAPH_BUILDER_MULTIGRAPH ,
    config_const.CONF_GRAPH_BUILDER_NORMALIZE_FEATURES]

    from datetime import datetime
    now = datetime.now()
    exp_name = now.strftime("%Y%m%d_%H%M%S")
    for conf in configs:
        item = params[conf]
        if isinstance(item, bool):
            item = int(item)
        elif isinstance(item, list):
            item = "_".join([i for i in item])
        exp_name+="_"+str(item)
    return exp_name    



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_env_info():
    logging.info(f"Environment details:")
    logging.info(f"Torch version: {torch.__version__}")
    logging.info(f"Torch geometric version: {torch_geometric.__version__}")
    logging.info(f"Spacy version: {spacy.__version__}")
    logging.info(f"Cuda available: {torch.cuda.is_available()}")
    logging.info(f"Cuda GPU count: {torch.cuda.device_count()}")
    logging.info(f"CPU count: {psutil.cpu_count()}")
    logging.info(f'Memory: total={convert_bytes(psutil.virtual_memory().total)}, available={convert_bytes(psutil.virtual_memory().available)}')

def convert_duration(type, duration):
    if type=="second":    
        minutes, seconds = divmod(duration, 60)
        hours, minutes = divmod(minutes, 60)
        return "%d:%02d:%02d" % (hours, minutes, seconds)
    return "Not supported ->" + type 

def convert_bytes(bytes_number):
    tags = [ "B", "KB", "MB", "GB", "TB" ]
 
    i = 0
    double_bytes = bytes_number
 
    while (i < len(tags) and  bytes_number >= 1024):
            double_bytes = bytes_number / 1024.0
            i = i + 1
            bytes_number = bytes_number / 1024
 
    return str(round(double_bytes, 2)) + " " + tags[i]

def calculate_metrics(y_pred, y_true, epoch, type, mlflow):
    print(f"\n Confusion matrix-{type}: \n {confusion_matrix(y_pred, y_true)}")
    print(f"F1 Score-{type}: {f1_score(y_pred, y_true)}")
    print(f"Accuracy-{type}: {accuracy_score(y_pred, y_true)}")
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f"Precision-{type}: {prec}")
    print(f"Recall-{type}: {rec}")
    if mlflow:
        mlflow.log_metric(key=f"Precision-{type}", value=float(prec), step=epoch)
        mlflow.log_metric(key=f"Recall-{type}", value=float(rec), step=epoch)
    try:
        roc = roc_auc_score(y_pred, y_true)
        print(f"ROC AUC-{type}: {roc}")
        if mlflow:
            mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(roc), step=epoch)
    except:
        print(f"ROC AUC-{type}: not defined")
        if mlflow:
            mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(0), step=epoch)

def is_numeric(string: str) -> bool:
    try:
        float(string)
        return True
    except ValueError:
        return False