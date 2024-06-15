from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, auc
import pandas as pd
import numpy as np
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

    def calculate_outliers_accuracy(self):
        self.outliers_accuracy = accuracy_score(
            self.true_labels[self.true_labels == 1], self.pred_labels[self.true_labels == 1]
        )
        return self.outliers_accuracy

    def calculate_precision(self):
        self.precision = precision_score(self.true_labels, self.pred_labels)
        return self.precision

    def calculate_recall(self):
        self.recall = recall_score(self.true_labels, self.pred_labels)
        return self.recall

    def calculate_f1(self):
        self.f1 = f1_score(self.true_labels, self.pred_labels)
        return self.f1

    def calculate_pr_curve(self):
        if self.scores is None:
            raise ValueError("scores is None. You can't calculate the pr curve.")
        precision, recall, _ = precision_recall_curve(self.true_labels, self.scores)
        return precision, recall

    def calculate_auc_pr(self):
        precision, recall = self.calculate_pr_curve()
        self.auc_pr = auc(recall, precision)
        return self.auc_pr

    def imbalanced_metrics(self):
        # Obliczanie macierzy konfuzji
        tn, fp, fn, tp = confusion_matrix(self.true_labels, self.pred_labels).ravel()

        # Obliczanie metryk
        positive_recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        negative_recall = tn / (tn + fp) if (tn + fp) != 0 else 0
        positive_precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        negative_precision = tn / (tn + fn) if (tn + fn) != 0 else 0

        # Obliczanie procentowej macierzy konfuzji
        total = tn + fp + fn + tp
        tp_percentage = tp / total
        fp_percentage = fp / total
        fn_percentage = fn / total
        tn_percentage = tn / total

        # Zwracanie metryk jako słownik
        metrics = {
            'positive_recall': positive_recall,
            'negative_recall': negative_recall,
            'positive_precision': positive_precision,
            'negative_precision': negative_precision,
            'tp_percentage': tp_percentage,
            'fp_percentage': fp_percentage,
            'fn_percentage': fn_percentage,
            'tn_percentage': tn_percentage
        }

        return metrics

    def calculate_all_metrics(self):
        metrics = {}

        metrics['accuracy'] = self.calculate_accuracy()
        metrics['outliers_accuracy'] = self.calculate_outliers_accuracy()
        metrics['precision'] = self.calculate_precision()
        metrics['recall'] = self.calculate_recall()
        metrics['f1'] = self.calculate_f1()

        if self.scores is not None:
            metrics['precision_recall_curve'] = self.calculate_pr_curve()
            metrics['auc_pr'] = self.calculate_auc_pr()
        else:
            metrics['precision_recall_curve'] = None
            metrics['auc_pr'] = None

        imbalanced_metrics = self.imbalanced_metrics()
        metrics.update(imbalanced_metrics)

        return metrics