""" Model validation metrics. Using Inferencer interface from tensorpack.
"""
from tensorpack.callbacks import Inferencer
from functional import seq
from sklearn.metrics import (average_precision_score, roc_auc_score)
from scipy.special import expit
import numpy as np

def _filter_all_negative(logits, label):
    """ AUC and mAP metrics only work when positive sample presents.
    However, in some batches, a class' ground truth labels could be all negative. 
    Thus, we need to filter out those classes to avoid computation error. 
    """
    keep_mask = np.any(label, axis=0)
    return logits[:, keep_mask], label[:, keep_mask]


def calcu_ap_auc(logits, labels):
    logits, labels = _filter_all_negative(logits, labels)
    prob = expit(logits)
    ap = average_precision_score(labels, prob)
    auc = roc_auc_score(labels, prob)
    return (ap, auc), prob.shape[0]


class Accumulator(object):
    def __init__(self, *args):
        self.names = args
        self.maintained_variables = [0 for _ in args]
        self.total_batches = 0
        
    def feed(self, batch_size, *args):
        assert len(self.maintained_variables) == len(args), \
            "you must feed the same number of variables as in initiation."
        
        divisor = self.total_batches + batch_size
        for i, value in enumerate(args):
            old = self.maintained_variables[i]
            new = (old * self.total_batches + value * batch_size) / divisor
            self.maintained_variables[i] = new
        self.total_batches += batch_size
        return self
        
    def retrive(self):
        ret = seq(self.names).zip(seq(self.maintained_variables)).dict()
        self.maintained_variables = [0 for _ in self.maintained_variables]
        self.total_batches = 0
        return ret

                                  
class ApAndAucScore(Inferencer):
    """ Calculate mAP, AUC and loss metrics using validation set.
    """

    def __init__(self):
        self.accu = Accumulator('val-average-precision', 'val-auc', 'val-loss')

    def _get_fetches(self):
        """ Required by the base class.

        Return: A list of tensor names.
        """
        return ['logits_export', 'label', 'loss_export']

    def _on_fetches(self, results):
        """ Required by the base class. Calculate metrics for a batch.
        """
        logits, label, loss = results
        (ap, auc), batch_size = calcu_ap_auc(logits, label)
        self.accu.feed(batch_size, ap, auc, loss)

    def _before_inference(self):
        """ Required by the base class. Clear stat before each epoch.
        """
        pass

    def _after_inference(self):
        """ Required by the base class. 

        Return: A dict of scalars for logging.
        """
        return self.accu.retrive()
