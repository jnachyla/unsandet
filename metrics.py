import numpy as np
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    precision_score,
    recall_score,
    accuracy_score,
)
from warnings import warn


class AnomalyDetectorEvaluator:
    def __init__(
        self,
        true_labels: np.ndarray = None,
        pred_labels: np.ndarray = None,
        scores=None,
    ):
        if true_labels is None or pred_labels is None:
            raise ValueError("true_labels and pred_labels must not be None.")
        if scores is None:
            warn("scores is None. You won't be able to calculate the pr curve.")

        if true_labels.shape != pred_labels.shape:
            raise ValueError("The shapes of true_labels and pred_labels are different.")

        self.true_labels = true_labels
        self.pred_labels = pred_labels
        self.scores = scores

    def calculate_accuracy(self):
        self.accuracy = accuracy_score(self.true_labels, self.pred_labels)
        return self.accuracy

    def calculate_precision(self):
        self.precision = precision_score(self.true_labels, self.pred_labels)
        return self.precision

    def calculate_recall(self):
        self.recall = recall_score(self.true_labels, self.pred_labels)
        return self.recall

    def calculate_pr_curve(self):
        if self.scores is None:
            raise ValueError("scores is None. You can't calculate the pr curve.")
        precision, recall, _ = precision_recall_curve(self.true_labels, self.scores)
        return precision, recall

    def calculate_auc_pr(self):
        precision, recall = self.calculate_pr_curve()
        self.auc_pr = auc(recall, precision)
        return self.auc_pr
