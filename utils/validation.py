from tensorpack.callbacks import Inferencer
from sklearn.metrics import (average_precision_score, roc_auc_score)
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class ApAndAucScore(Inferencer):
     
    def _get_fetches(self):
        return ['logits_export', 'label', 'loss_export']
    
    def _on_fetches(self, results):
        logits, label, loss = results
        logits, label = self._filter_all_negative(logits, label)
        prob = sigmoid(logits)
        ap = average_precision_score(label, prob)
        auc = roc_auc_score(label, prob)
        size = prob.shape[0]
        self._moving_ap = (self._moving_ap * self._last_size + ap * size) / \
                               (self._last_size + size)
        self._moving_auc = (self._moving_auc * self._last_size + auc * size) / \
                               (self._last_size + size)
        self._moving_loss = (self._moving_loss * self._last_size + loss * size) / \
                               (self._last_size + size)
        self._last_size += size
        
    def _filter_all_negative(self, logits, label):
        keep_mask = np.any(label, axis=0)
        return logits[:, keep_mask], label[:, keep_mask]
    
    def _before_inference(self):
        self._moving_ap = 0.0
        self._moving_auc = 0.0
        self._moving_loss = 0.0
        self._last_size = 0
    
    def _after_inference(self):
        return { 'val-average-precision': self._moving_ap, 
                 'val-auc-score': self._moving_auc, 
                 'val-loss': self._moving_loss }
    