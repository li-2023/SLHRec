import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import log_loss
import random

def auc_acc(labels, pred):
    auc = roc_auc_score(labels, pred)
    predictions = [1 if i >= 0.5 else 0 for i in pred]
    acc = np.mean(np.equal(predictions, labels))
    
    scores = pred
    scores[scores >= 0.5] = 1
    scores[scores < 0.5] = 0
    
    f1 = f1_score(y_true=labels, y_pred=scores)
    
    return auc, acc, f1
