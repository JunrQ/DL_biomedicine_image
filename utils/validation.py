""" Model validation metrics. Using Inferencer interface from tensorpack.
"""
from tensorpack.callbacks import Inferencer
from sklearn.metrics import (average_precision_score, roc_auc_score)
import numpy as np


def sigmoid(x):
    """ Simple sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))


class ApAndAucScore(Inferencer):
    """ Calculate mAP, AUC and loss metrics using validation set.
    """

    def __init__(self):
        self._moving_ap = 0.0
        self._moving_auc = 0.0
        self._moving_loss = 0.0
        self._last_size = 0

    def _get_fetches(self):
        """ Required by the base class.

        Return: A list of tensor names.
        """
        return ['logits_export', 'label', 'loss_export']

    def _on_fetches(self, results):
        """ Required by the base class. Calculate metrics for a batch.
        """
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
        """ AUC and mAP metrics only work when positive sample presents.
        However, in some batches, a class' ground truth labels could be all negative. 
        Thus, we need to filter out those classes to avoid computation error. 
        """
        keep_mask = np.any(label, axis=0)
        return logits[:, keep_mask], label[:, keep_mask]

    def _before_inference(self):
        """ Required by the base class. Clear stat before each epoch.
        """
        self._moving_ap = 0.0
        self._moving_auc = 0.0
        self._moving_loss = 0.0
        self._last_size = 0

    def _after_inference(self):
        """ Required by the base class. 

        Return: A dict of scalars for logging.
        """
        return {'val-average-precision': self._moving_ap,
                'val-auc-score': self._moving_auc,
                'val-loss': self._moving_loss}
