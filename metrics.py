import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    precision_score,
    recall_score,
    accuracy_score, confusion_matrix,
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

        # Tworzenie DataFrame z procentową macierzą konfuzji
        data_percentage = {
            '': ['Positive Class', 'Negative Class'],
            'Positive Prediction': [f'True Positive (TP) {tp_percentage:.2%}',
                                    f'False Positive (FP) {fp_percentage:.2%}'],
            'Negative Prediction': [f'False Negative (FN) {fn_percentage:.2%}',
                                    f'True Negative (TN) {tn_percentage:.2%}']
        }

        df_percentage = pd.DataFrame(data_percentage)

        # Tworzenie DataFrame z metrykami procentowymi
        metrics_data = {
            'Metric': ['Positive Recall', 'Negative Recall', 'Positive Precision', 'Negative Precision'],
            'Value': [
                f'{positive_recall:.2%}',
                f'{negative_recall:.2%}',
                f'{positive_precision:.2%}',
                f'{negative_precision:.2%}'
            ]
        }

        df_metrics = pd.DataFrame(metrics_data)

        return df_percentage,df_metrics

    def calculate_all_metrics(self):
        metrics = {}

        metrics['accuracy'] = self.calculate_accuracy()
        metrics['outliers_accuracy'] = self.calculate_outliers_accuracy()
        metrics['precision'] = self.calculate_precision()
        metrics['recall'] = self.calculate_recall()

        if self.scores is not None:
            metrics['precision_recall_curve'] = self.calculate_pr_curve()
            metrics['auc_pr'] = self.calculate_auc_pr()
        else:
            metrics['precision_recall_curve'] = None
            metrics['auc_pr'] = None

        confusion_matrix_percentage_df, metrics_df = self.imbalanced_metrics()
        metrics['confusion_matrix_percentage'] = confusion_matrix_percentage_df
        metrics['imbalanced_metrics'] = metrics_df

        return metrics