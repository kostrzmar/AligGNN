"""
from word alignment by reusing Paraller Phrases by Maria Holmqvist - 2008

from Michalcena & Pedersen 2003
precision -> measures the proportion of correct links in the computed alignments (A) and the set of golden
                standard alignment (G)
p(A,G) = len(G and A) / len(A)

recall -> measures the proportion of correctly computed links in the set of golden standard links 
r(A,G) = len(G And A) / len(G)

f_1(P,R) = (2*P*R) / (P+R)

from Och and Ney (2003) - introducted two type of alignments: S(sure), P(possible) -> S is in P...

precision error occurs only if computed link is not even a possible link in gold standard

p(A, P)= len(P and A) / len(P)

recall error occurs only with sure links 
r(A, S) = len(S and A) / len(S)

Alignment error rate AER = 1 -   ( len(S and A)  + len(P and A)) / (len(S) + len(A)

"""


def evaluate_f1(sets, predict, gold):
    tp_count = 0
    fp_count = 0

    fn_count = len(gold)
    tn_count = len(sets) - len(gold)

    _accuracy = 0
    _recall = 0
    _precision = 0
    _f1 = 0

    for item in predict:
        if item in gold:
            tp_count+=1
            fn_count-=1
        else:
            fp_count+=1
            tn_count-=1
    _accuracy = (tp_count + tn_count)/(tp_count + tn_count + fp_count + fn_count)
    if tp_count + fn_count > 0:
        _recall = tp_count/(tp_count + fn_count)
    
    if tp_count + fp_count >0:
        _precision = tp_count/(tp_count + fp_count)
    
    if _precision + _recall>0:
        _f1  = 2 * _precision * _recall / (_precision + _recall)
    return _accuracy, _precision, _recall, _f1


def intrinsic_valuation(predict, gold):
    _recall = 0
    _precision = 0
    _f1 = 0
    _aer = 0
    count = 0
    for item in predict:
        if item in gold: 
            count +=1
    if len(predict) > 0:
        _precision = count / len(predict)
    if len(gold) >0:
        _recall = count / len(gold)
    if _precision+_recall>0:    
        _f1 = (2*_precision*_recall)/(_precision+_recall)
    if len(predict) + len(gold) > 0:
        _aer = 1 - ((2*count)/(len(predict)+len(gold)))
    return _precision, _recall, _f1, _aer

def intrinsic_valuation_sure_possible(predict, goldSure, goldPossible):
    countP = 0
    countS = 0
    for item in predict:
        if item in goldSure: 
            countS +=1
        if item in goldPossible:
            countP +=1
    _precision = countP / len(goldPossible)
    _recall = countS / len(goldSure)
    _aer = 1 - ((countS + countP)/(len(goldSure)+len(predict)))
    return _precision, _recall, None, _aer    