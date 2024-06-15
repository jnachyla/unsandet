import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import data_prep
from ad_meta_cost import MetaCost, generate_probability_dependent_cost_matrix, generate_fixed_interval_cost_matrix, AnomalyDetector
from ad_one_class import OneClassAnnomalyDetector

from metrics import AnomalyDetectorEvaluator
from subsampler import subsample_majority_class


class Experiments:
    def __init__(self):
        self.http_dataset = self.load_data_http()
        self.shuttle_dataset = self.load_data_shuttle()

    def load_data_http(self):


        X = pd.read_csv("http_train.csv",header=0).values
        y = pd.read_csv("http_eval.csv",header=0).values

        return (X,y.ravel())



    def load_data_shuttle(self):
        X = pd.read_csv("shuttle_train.csv",header=0).values
        y = pd.read_csv("shuttle_eval.csv",header=0).values

        return (X,y.ravel())

    def _scale(self, X,y):
        scaler = StandardScaler()
        X = scaler.fit_transform(X = X, y = y)
        return (X,y)

    def run_http_one_class(self):
        """
        Run one class anomaly detection on the http dataset.
        :return:
        """
        X,y = self.http_dataset

        (X,y) = subsample_majority_class(X, y, fraction=0.4)

        (X,y) = self._scale(X, y)

        Xtrain, ytrain, Xtest, ytest = data_prep.split_binary_dataset(X, y, inliers_to_outliers_ratio=3.0)

        #fit isolation forest
        print("Fitting Isolation Forest...")
        #print shape of Xtrain with name formated
        print(f"Shape of Xtrain: {Xtrain.shape}")
        isolation_forest = OneClassAnnomalyDetector(model_name = "isolationforest")
        isolation_forest.fit(Xtrain)
        print("Fitted. Predicting...")
        ypred_forest = isolation_forest.predict(Xtest)

        print("Results: HTTP one class IsolationForest")

        evaluator_http_forest = AnomalyDetectorEvaluator(true_labels=ytest, pred_labels=ypred_forest, scores=None)
        print(evaluator_http_forest.calculate_all_metrics())

        svm = OneClassAnnomalyDetector(model_name = "oneclasssvm")
        print("Fitting Model...")

        svm.fit(Xtrain)
        print("Fitted. Predicting...")
        ypred_svm = svm.predict(Xtest)

        print("Results: HTTP one class SVM")
        evaluator_http_forest = AnomalyDetectorEvaluator(true_labels=ytest, pred_labels=ypred_svm, scores=None)
        print(evaluator_http_forest.calculate_all_metrics())

    def _evaluate_meta_cost(self,X, y, cost_matrix_generator="fixed_interval", n = 1000, m = 30, N=2, cost_matrix=None):
        """
        Ocena modelu MetaCost na zbiorze treningowym i testowym.

        Parametry:
        X: np.ndarray
            Zbiór danych do trenowania modelu.
        y: np.ndarray
            Etykiety danych.
        cost_matrix_generator: str lub function
            Metoda generowania macierzy kosztów.
        N: int
            Liczba iteracji oceny.
        cost_matrix: np.ndarray
            Własna macierz kosztów (opcjonalnie).

        Zwraca:
        dict: Średnie wartości metryk Precision, Recall, F1 Score oraz Accuracy.
        """

        all_metrics = []

        for _ in tqdm(range(N)):
            class_probabilities = np.bincount(y) / len(y)
            if cost_matrix is not None:
                cost_matrix = cost_matrix
            elif cost_matrix_generator == "fixed_interval":
                cost_matrix = generate_fixed_interval_cost_matrix(2)
            else:
                cost_matrix = generate_probability_dependent_cost_matrix(2, class_probabilities)

            detector = AnomalyDetector(n_clusters=2, metric="euclidean", model_name="kmeans")

            print("Meta Cost n samples: ", n)
            meta_cost = MetaCost(base_detector=detector, cost_matrix=cost_matrix, m=m, n=n)

            # Trenowanie modelu i przypisanie klastrów
            assigned_clusters = meta_cost.fit_predict(X, y)

            # Mapowanie klastrów na oryginalne klasy (zakładamy, że klasa 0 jest dominująca)
            y_pred = AnomalyDetector.transform_labels(assigned_clusters)

            evaluator = AnomalyDetectorEvaluator(true_labels=y, pred_labels=y_pred)

            metrics = evaluator.calculate_all_metrics()
            all_metrics.append(metrics)

        avg_metrics = {}
        for metric in all_metrics[0].keys():

            try:
                if metric in ['precision_recall_curve', 'confusion_matrix_percentage']:
                    continue


                metric_values = []
                for metrics in all_metrics:
                    metric_value = metrics.get(metric)
                    if metric_value is not None and metric_value != 0:
                        metric_values.append(metric_value)

                if metric_values:
                    avg_metrics[metric] = np.mean(metric_values)
                else:
                    avg_metrics[metric] = None
            except:
                continue
        return avg_metrics
    def run_shuttle_meta_cost(self):
        """
        Run MetaCost on the shuttle dataset.
        :return:
        """
        X,y = self.shuttle_dataset

        print("Evaluating MetaCost on the Shuttle dataset...")
        cost_matrix = np.array([[0, 1],
                                [10, 0]])
        avg_metrics = self._evaluate_meta_cost(X, y, cost_matrix_generator="fixed_interval", n=1000, m=30, N=2, cost_matrix=cost_matrix)

        print("Results: Shuttle MetaCost")
        print(avg_metrics)
    def run_http_kmeans(self):
        X = self.http_dataset[0]
        y = np.ravel(self.http_dataset[1])


        kmeans = AnomalyDetector(model="kmeans", n_clusters=2)
        pred_kmeans, distances_kmeans = kmeans.fit_predict(data=X)
        labels_kmeans = AnomalyDetector.transform_labels(pred_kmeans)
        distances_shuttle_kmeans = AnomalyDetector.transform_distances(labels_kmeans)
        evaluator = AnomalyDetectorEvaluator(y, pred_kmeans, distances_kmeans)

        print(evaluator.calculate_all_metrics())

    def run_http_kmeans_metacost(self):
        X = self.http_dataset[0]
        y = np.ravel(self.http_dataset[1])


        kmeans = AnomalyDetector(model="kmeans", n_clusters=2)
        cost_matrix = np.array([[0, 1], [1, 0]])

        meta_cost = MetaCost(base_detector=kmeans, cost_matrix=cost_matrix, m=3, n=1000)
        y_predicted = meta_cost.fit_predict(X)


        labels_kmeans = AnomalyDetector.transform_labels(y_predicted)

        evaluator = AnomalyDetectorEvaluator(y, labels_kmeans, None)

        print(evaluator.calculate_all_metrics())

exps = Experiments()
exps.run_shuttle_meta_cost()


