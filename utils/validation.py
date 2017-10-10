from tensorpack.callbacks import Inferencer
from sklearn.metrics import (average_precision_score, roc_auc_score)
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class ApAndAucScore(Inferencer):
    def __init__(self):
        self._moving_ap = 0.0
        self._moving_auc = 0.0
        self._last_size = 0
        
    def _get_fetches(self):
        return ['logits_export', 'label']
    
    def _on_fetches(self, results):
        logits, label= results
        prob = sigmoid(logits)
        ap = average_precision_score(label, prob)
        auc = roc_auc_score(label, prob)
        size = prob.shape[0]
        self._moving_ap = (self._moving_ap * self._last_size + ap * size) / \
                               (self._last_size + size)
        self._moving_auc = (self._moving_auc * self._last_size + auc * size) / \
                               (self._last_size + size)        
        self._last_size = size
    
    def _after_inference(self):
        return { 'average-precision-score': self._moving_ap, 
                 'auc-score': self._moving_auc }
    