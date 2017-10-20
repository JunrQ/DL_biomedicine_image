""" Model validation metrics. Using Inferencer interface from tensorpack.
"""
from tensorpack.callbacks import Inferencer
from functional import seq
from sklearn import metrics
from sklearn.preprocessing import binarize
from scipy.special import expit
import numpy as np


def _filter_all_negative(logits, label):
    """ AUC and mAP metrics only work when positive sample presents.
    However, in some batches, a class' ground truth labels could be all negative. 
    Thus, we need to filter out those classes to avoid computation error. 
    """
    keep_mask = np.any(label, axis=0)
    return logits[:, keep_mask], label[:, keep_mask]



def pred_from_score(scores, threshold):
    """ Convert real number score (aka. confidence) to 0 or 1.

    Args:
        scores: Real number scores, ndarray of shape [N, L].
        threshold: Threshold of true prediction. A scalar or
            an array of length L for a per-label threshold.
    """
    cut = scores - threshold
    return binarize(cut, threshold=0.0, copy=False)


def calcu_one_metric(scores, labels, metric, threshold=None):
    ans = None

    if metric == 'mean_average_precision':
        scores, labels = _filter_all_negative(scores, labels)
        ans = metrics.average_precision_score(labels, scores)

    elif metric == 'macro_auc':
        scores, labels = _filter_all_negative(scores, labels)
        ans = metrics.roc_auc_score(labels, scores, average='macro')

    elif metric == 'micro_auc':
        scores, labels = _filter_all_negative(scores, labels)
        ans = metrics.roc_auc_score(labels, scores, average='micro')

    elif metric == 'macro_f1':
        scores, labels = _filter_all_negative(scores, labels)
        pred = pred_from_score(scores, threshold)
        ans = metrics.f1_score(labels, pred, average='macro')

    elif metric == 'micro_f1':
        scores, labels = _filter_all_negative(scores, labels)
        pred = pred_from_score(scores, threshold)
        ans = metrics.f1_score(labels, pred, average='micro')

    elif metric == 'rank_mean_average_precision':
        ans = metrics.label_ranking_average_precision_score(labels, scores)

    elif metric == 'coverage':
        cove = metrics.coverage_error(labels, scores)
        # see http://scikit-learn.org/stable/modules/model_evaluation.html#coverage-error
        ans = cove - 1

    elif metric == 'rank_loss':
        ans = metrics.label_ranking_loss(labels, scores)

    elif metric == 'one_error':
        top_score = np.argmax(scores, axis=1)
        top_label = labels[:, top_score]
        ans = 1 - np.sum(top_label) / len(top_label)

    else:
        raise f"unsuppored metric: {metric}"

    return ans


def calcu_metrics(logits, labels, queries):
    scores = expit(logits)
    ans = seq(queries).map(
        lambda m: calcu_one_metric(scores, labels, m)).list()
    return ans


class Accumulator(object):
    """ Maintain moving average for multiple values.
    """
    def __init__(self, *args):
        self.names = args
        self.maintained_variables = [0 for _ in args]
        self.total_batches = 0

    def feed(self, batch_size, *args):
        """ Feed accumulator a new batch.
        """
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
        """ Get moving average and clear memory.
        """
        ret = seq(self.names).zip(seq(self.maintained_variables)).dict()
        self.maintained_variables = [0 for _ in self.maintained_variables]
        self.total_batches = 0
        return ret


class AggerateMetric(Inferencer):
    """ Calculate mAP, AUC and loss metrics using validation set.
    """

    def __init__(self, queries):
        self.queries = queries
        self.accu = Accumulator(*queries)

    def _get_fetches(self):
        """ Required by the base class.

        Return: A list of tensor names.
        """
        return ['logits_export', 'label']

    def _on_fetches(self, results):
        """ Required by the base class. Calculate metrics for a batch.
        """
        logits, labels= results
        batch_size = logits.shape[0]
        new_values = calcu_metrics(logits, labels, self.queries)
        self.accu.feed(batch_size, *new_values)

    def _before_inference(self):
        """ Required by the base class. Clear stat before each epoch.
        """
        pass

    def _after_inference(self):
        """ Required by the base class. 

        Return: A dict of scalars for logging.
        """
        return self.accu.retrive()
