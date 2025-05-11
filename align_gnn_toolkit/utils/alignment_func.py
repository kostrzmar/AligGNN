import torch
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pandas as pd
from utils import  alignment_eval_metric as alignment_eval_metric 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from utils import utils_trained_models

def show_diagram(f_p, f_r, f_f1, _range ):

    fig, ax = plt.subplots(figsize=(10,4))
    sns.lineplot(y=gaussian_filter1d(f_p, sigma=1), x=_range, ax=ax, label="Precision", linestyle="-")
    sns.lineplot(y=gaussian_filter1d(f_r, sigma=1), x=_range, ax=ax, label="Recall", linestyle="-")
    sns.lineplot(y=gaussian_filter1d(f_f1, sigma=0.4), x=_range, ax=ax, label="F1", linestyle="-")
    #sns.lineplot(y=f_f1, x=_range, ax=ax, label="F1", linestyle="-")

    mode_idx = np.argmax(f_f1)
    ax.vlines(_range[mode_idx], 0, f_f1[mode_idx], ls='--', color="red")
    plt.xlabel('threshold', fontsize=20)
    plt.ylabel('score', fontsize=20)
    plt.title(f'F1 maximum {f_f1[mode_idx]:.2f}  for threshold:{_range[mode_idx]:.2f}', loc='left', color='blue', size=20)
    return fig
    
    
def read_ds(path_to_ds):
    file = open(path_to_ds, 'r', encoding='utf-8')
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
    return source, target, aligned    



import numpy as np
def extract_alignments(all_sim_scores, all_gold, threshold):
    focusCube = torch.Tensor(all_sim_scores)
    #goldCube = torch.Tensor([ 1 if x == "aligned" else 0 for x in all_gold])
    goldCube = torch.Tensor(all_gold)
    mask = torch.ones(focusCube.size())
    alignments = torch.nonzero((focusCube >= threshold)*mask).tolist()
    golds = torch.nonzero((goldCube >= threshold)*mask).tolist()
    return alignments, golds
  

def evaluate_alignments_in_range( all_sim_scores, all_gold, show_progress=False):
    f_p = []
    f_r = []
    f_f1 = []
    _range = np.arange(0.05, 1, 0.01).tolist()
    for thr in tqdm(_range):
        alignments, golds = extract_alignments(all_sim_scores, all_gold, thr)
        p,r,f1,aer = alignment_eval_metric.intrinsic_valuation(alignments, golds)
        f_p.append(p)
        f_r.append(r)
        f_f1.append(f1)
    mode_idx = np.argmax(f_f1)
    return f_p, f_r, f_f1, _range, mode_idx


import math

def insertChar(mystring, position, chartoinsert ):
    mystring   =  mystring[:position] + chartoinsert + mystring[position:] 
    return mystring  

def split_long_per_line(str, at_column):
    split_count = math.ceil(len(str)/at_column)
    for i in range(1,split_count):
        str = insertChar(str, at_column*i, "\n")
    return str


def display_evidence(item, all_sim_scores, all_gold, _range, mode_idx,   should_print=True, new_line_column=None, return_as_list=False, test_loader=None):
    if return_as_list:
        sentences = utils_trained_models.get_sentences([item], test_loader, all_sim_scores, all_gold, as_list=True)
        return sentences[0][0], sentences[0][1]
        #return test_lsents[item], test_rsents[item]
    else:
        sentences = utils_trained_models.get_sentences([item], test_loader, all_sim_scores, all_gold, as_list=True)
        out = f'Id: {item} Label: {all_gold[item]} Score: {all_sim_scores[item]:.3f}, Thr {_range[mode_idx]:.3f}\n' 
        if new_line_column:
            out += f'S: {split_long_per_line(sentences[0][0], new_line_column)}\n'
            out += f'T: {split_long_per_line(sentences[0][1], new_line_column)}\n'
        else:
            out += f'S: {sentences[0][0]}\n'
            out += f'T: {sentences[0][1]}\n'
        out +='\n'
        if should_print:
            print(out)
        else:
            return out

def display_errors(category, how_many, title):
    print(title)
    for item in list(category)[:how_many]:
        display_evidence(item, all_sim_scores, _range, mode_idx, test_lsents, test_rsents)

        


def get_text_page(content,subject,size):
    page = plt.figure(figsize=size)
    page.clf()
    page.text(0.05, 0.95, subject, fontsize=12,  ha='left', va='top', transform=page.transFigure)
    page.text(0.05, 0.90, content, fontsize=11,  ha='left', va='top', transform=page.transFigure)
    return page

def get_report_for_evidences(pdf, category, category_name, all_sim_scores, all_gold, _range, mode_idx,  bins=10, max_examples=7, new_line_column=110, test_loader=None):
    DO_ALIGNMENT_ANALYSIS = False
    bins = np.histogram_bin_edges([all_sim_scores[item] for item in list(category)], bins)
    bins = np.append(bins, 1.0)
    prev = -1
    current =0
    results_per_bin = {}
    for index, bin in enumerate(bins):
        current=bin
        bin_str = f'{prev:.3f}-{current:.3f}'
        results_per_bin[bin_str] = {}
        all_items = [item for item in list(category) if all_sim_scores[item] > prev and all_sim_scores[item] <current ]
        items_to_display = all_items[:max_examples]
        items_to_analyse = random.sample(all_items, min(len(all_items),100))
        evidences = ""
        for item in items_to_display:
            #evidences += display_evidence(item, all_sim_scores, all_gold, _range, mode_idx, test_lsents, test_rsents, labels, should_print=False, new_line_column=new_line_column)
            evidences += display_evidence(item, all_sim_scores, all_gold, _range, mode_idx, should_print=False, new_line_column=new_line_column, test_loader=test_loader)
        if len(evidences) > 1:
            pdf.savefig(get_text_page(evidences, f'[{category_name}] Evidence between {prev:.3f} and {current:.3f}', (10,10)))
            plt.close()
        
        if DO_ALIGNMENT_ANALYSIS:
            if len(items_to_analyse)>0:
                        
                ds = Dataset("Test")
                source, target = [], []
                for item in items_to_analyse:
                    src, trg = display_evidence(item, all_sim_scores,all_gold, _range, mode_idx, should_print=False, new_line_column=new_line_column, return_as_list=True,test_loader=test_loader)
                    source.append(src) 
                    target.append(trg)
                results_per_bin[bin_str] = ds.getStatisticAsString(source, target, return_as_str=False)
                #pdf.savefig(get_text_page(ds.getStatisticAsString(source, target), f'Stats between {prev:.2f} and {current:.2f} \n', (10,10)))
                #plt.close()
        prev=current    
    
    output = {}
    has_value = False
    for bin in results_per_bin:
        output[bin] = {}
        for prop in results_per_bin[bin]:
            output[bin][prop] = f'{results_per_bin[bin][prop]["mean"]:.2f}+/-{results_per_bin[bin][prop]["std"]:.2f} [{results_per_bin[bin][prop]["min"]:.2f}¦{results_per_bin[bin][prop]["max"]:.2f}]'  
            has_value = True
    if has_value:
        pdf.savefig(get_table(output))
        plt.close()
    
    if DO_ALIGNMENT_ANALYSIS:
        x_range = [x.split("-")[0] for x in  results_per_bin.keys() if x.split("-")[0] !='']
        get_negatives = [x for x in  results_per_bin.keys() if x.split("-")[0] =='']
        for i in range(len(get_negatives)):
            x_range.insert(0, 0.00)
        diagrams = [
            (["tokens_src", "tokens_trg"], "nbr of tokens", "score distribution" ),
            (["tokens_src_length", "tokens_trg_length"], "length of tokens", "score distribution"),
            (["sbert_similarity", "Levenshtein similarity"], "similarity score", "score distribution"),
            (["unigrams", "bigrams","trigrams"],"count", "score distribution" ),
            (["Additions proportion", "Deletions proportion", "Sentence splits"],"count", "score distribution"),
            (["Compression ratio", "Lexical complexity score"],"score", "score distribution"),
            (["fkgl_s", "fkgl_t"],"score", "score distribution"),
            (["flesch_reading_ease_src", "flesch_reading_ease_trg"],"score", "score distribution"),
            (["dale_chall_readability_score_src", "dale_chall_readability_score_trg"],"score", "score distribution"),
        ] 
        for diagram in diagrams:
            pdf.savefig(plot_diagram(diagram, results_per_bin, x_range))
            plt.close()
    
    return results_per_bin


def get_stats(metric, stats):
    y_s = []
    for range in stats:
        if metric in stats[range]:
            y_s.append(stats[range][metric]["mean"])
        else:
            y_s.append(np.inf)
    return y_s

def plot_diagram(diagram, stats, x_range):
    fig, ax = plt.subplots(figsize=(10,6))
    for metric in diagram[0]:
        sns.lineplot(y=get_stats(metric, stats), x=x_range, ax=ax, label=metric, linestyle="-")
    plt.xlabel(diagram[2], fontsize=20)
    plt.ylabel(diagram[1], fontsize=20)
    return fig

def get_table(output):
    df = pd.DataFrame(output)
    fig, ax =plt.subplots(figsize=(20,10))
    ax.axis('tight')
    ax.axis('off')
    ax.table(rowLabels=df.index,cellText=df.values,colLabels=df.columns,loc='center')
    return fig


def get_histogram(all_sim_scores):
    fig, ax = plt.subplots(figsize=(10,8))
    sns.histplot(all_sim_scores, bins=50, kde=True, color='skyblue', edgecolor='red', ax=ax)
    
    # Adding labels and title
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.title('Score distribution')
    return fig
