import json

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

    def run_http_one_class_anomaly_detection(self):
        """
        Run one class anomaly detection on the HTTP dataset and save results to JSON files.
        :return:
        """
        X, y = self.http_dataset

        (X, y) = subsample_majority_class(X, y, fraction=0.4)
        (X, y) = self._scale(X, y)

        Xtrain, ytrain, Xtest, ytest = data_prep.split_binary_dataset(X, y, inliers_to_outliers_ratio=3.0)

        # Isolation Forest
        print("Fitting Isolation Forest...")
        print(f"Shape of Xtrain: {Xtrain.shape}")
        isolation_forest = OneClassAnnomalyDetector(model_name="isolationforest")
        isolation_forest.fit(Xtrain)
        print("Fitted. Predicting...")
        ypred_forest = isolation_forest.predict(Xtest)

        evaluator_http_forest = AnomalyDetectorEvaluator(true_labels=ytest, pred_labels=ypred_forest, scores=None)
        forest_metrics = evaluator_http_forest.calculate_all_metrics()
        print("Results: HTTP one class IsolationForest")
        print(forest_metrics)

        # Save results to JSON
        with open('http_isolation_forest_results.json', 'w') as f:
            json.dump(forest_metrics, f, indent=4)

        # One-Class SVM
        svm = OneClassAnnomalyDetector(model_name="oneclasssvm")
        print("Fitting One-Class SVM...")
        svm.fit(Xtrain)
        print("Fitted. Predicting...")
        ypred_svm = svm.predict(Xtest)

        evaluator_http_svm = AnomalyDetectorEvaluator(true_labels=ytest, pred_labels=ypred_svm, scores=None)
        svm_metrics = evaluator_http_svm.calculate_all_metrics()
        print("Results: HTTP one class SVM")
        print(svm_metrics)

        # Save results to JSON
        with open('http_one_class_svm_results.json', 'w') as f:
            json.dump(svm_metrics, f, indent=4)

        print("Results saved to JSON files")

    def _evaluate_meta_cost(self,X, y,cost_matrix_generator="fixed_interval", n = 1000, m = 30, N=2, cost_matrix=None, detector = None):
        """
        Ocena modelu MetaCost

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

        if detector is None:
            raise ValueError("Annomaly Detector must be provided.")

        all_metrics = []

        for _ in tqdm(range(N)):
            class_probabilities = np.bincount(y) / len(y)
            if cost_matrix is not None:
                cost_matrix = cost_matrix
            elif cost_matrix_generator == "fixed_interval":
                cost_matrix = generate_fixed_interval_cost_matrix(2)
            else:
                cost_matrix = generate_probability_dependent_cost_matrix(2, class_probabilities)

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
        Run MetaCost on the Shuttle dataset with varying distance metrics and save results to a JSON file.
        :return:
        """
        X = self.shuttle_dataset[0]
        y = np.ravel(self.shuttle_dataset[1])
        y = y.astype(np.int64)

        print("Evaluating MetaCost on the Shuttle dataset...")

        # Definiowanie metryk odległości
        distance_metrics = ["mahalanobis","euclidean", "cityblock"]
        n = 1000
        m = 30
        N = 5

        all_results = []

        for metric in distance_metrics:
            print(f"Running MetaCost with metric={metric}...")

            # Create setted based on experiments
            cost = 10
            cost_matrix = np.array([[0, 1],
                                    [cost, 0]])

            detector = AnomalyDetector(n_clusters=2, metric=metric, model_name="kmeans")
            avg_metrics = self._evaluate_meta_cost(X, y, cost_matrix_generator="fixed_interval", n=n, m=m, N=N,
                                                   cost_matrix=cost_matrix, detector=detector)

            print(f"MetaCost results with metric={metric}: {cost_matrix}")
            print(avg_metrics)

            result = {
                'metric': metric,
                'cost_matrix': cost_matrix.tolist(),
                'avg_metrics': avg_metrics,
                'n': n,
                'm': m,
                'N': N
            }
            all_results.append(result)

        # Zapis wyników do pliku JSON
        with open('meta_cost_shutle_results.json', 'w') as f:
            json.dump(all_results, f, indent=4)

        print("Results saved to meta_cost_results.json")


    def run_http_meta_cost(self):
        """
        Run MetaCost on the HTTP dataset with varying distance metrics and save results to a JSON file.
        :return:
        """
        X = self.http_dataset[0]
        y = np.ravel(self.http_dataset[1])
        y = y.astype(np.int64)

        print("Evaluating MetaCost on the HTTP dataset...")

        # Definiowanie metryk odległości
        distance_metrics = ["mahalanobis","euclidean", "cityblock"]
        n = 10_000  # Stała wartość n
        m = 30  # Stała wartość m
        N = 5  # Liczba iteracji oceny

        all_results = []

        for metric in distance_metrics:
            print(f"Running MetaCost with metric={metric}...")

            # Create cost matrix based on disproportion of minority/majority class
            cost = 1 + ((np.bincount(y)[1] / np.bincount(y)[0]))
            cost_matrix = np.array([[0, 1],
                                    [cost, 0]])

            detector = AnomalyDetector(n_clusters=2, metric=metric, model_name="kmeans")
            avg_metrics = self._evaluate_meta_cost(X, y, cost_matrix_generator="fixed_interval", n=n, m=m, N=N,
                                                   cost_matrix=cost_matrix, detector=detector)

            print(f"MetaCost results with metric={metric}: {cost_matrix}")
            print(avg_metrics)

            result = {
                'metric': metric,
                'cost_matrix': cost_matrix.tolist(),
                'avg_metrics': avg_metrics,
                'n': n,
                'm': m,
                'N': N
            }
            all_results.append(result)

        # Zapis wyników do pliku JSON
        with open('meta_cost_http_results.json', 'w') as f:
            json.dump(all_results, f, indent=4)

        print("Results saved to meta_cost_results.json")



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
exps.run_http_one_class_anomaly_detection()


