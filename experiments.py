import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score

import data_prep
from ad_one_class import OneClassAnnomalyDetector
from anomaly_detector import AnomalyDetector
from metrics import AnomalyDetectorEvaluator


class Experiments:
    def __init__(self):
        self.http_dataset = self.load_data_http()
        self.shuttle_dataset = self.load_data_shuttle()
        pass
    def load_data_http(self):


        X = pd.read_csv("http_train.csv",header=0).values
        y = pd.read_csv("http_eval.csv",header=0).values

        return (X,y)



    def load_data_shuttle(self):
        X = pd.read_csv("shuttle_train.csv",header=0).values
        y = pd.read_csv("shuttle_eval.csv",header=0).values

        return (X,y)


    def run_http_one_class(self):
        """
        Run one class anomaly detection on the http dataset.
        TODO: jedna metoda run i przyjmuje model jako argument, X,y jako argument, zwraca wyniki
        :return:
        """
        Xtrain, ytrain, Xtest, ytest = data_prep.split_binary_dataset(self.http_dataset[0], self.http_dataset[1])

        #fit isolation forest
        print("Fitting Isolation Forest...")
        #print shape of Xtrain with name formated
        print(f"Shape of Xtrain: {Xtrain.shape}")
        isolation_forest = OneClassAnnomalyDetector(model_name = "isolationforest")
        isolation_forest.fit(Xtrain)
        print("Fitted. Predicting...")
        ypred_forest = isolation_forest.predict(Xtest)
        print("Results: HTTP one class IsolationForest")
        self.print_metrics(ytest, ypred_forest)

        svm = OneClassAnnomalyDetector(model_name = "oneclasssvm")
        print("Fitting Model...")

        svm.fit(Xtrain)
        print("Fitted. Predicting...")
        ypred_svm = svm.predict(Xtest)

        print("Results: HTTP one class SVM")
        self.print_metrics(ytest, ypred_svm)
    def run_http_kmeans(self):
        X = self.http_dataset[0]
        y = np.ravel(self.http_dataset[1])

        kmeans = AnomalyDetector(model="kmeans", n_clusters=2)
        labels_shuttle_kmeans, distances_shuttle_kmeans = kmeans.fit_predict(data=X)
        labels_shuttle_kmeans = AnomalyDetector.transform_labels(labels_shuttle_kmeans)
        distances_shuttle_kmeans = AnomalyDetector.transform_distances(distances_shuttle_kmeans)

        evaluator_shuttle_kmeans = AnomalyDetectorEvaluator( y,labels_shuttle_kmeans,distances_shuttle_kmeans)
        accuracy_shuttle_kmeans = evaluator_shuttle_kmeans.calculate_accuracy()
        recall_shuttle_kmeans = evaluator_shuttle_kmeans.calculate_recall()
        precision_shuttle_kmeans = evaluator_shuttle_kmeans.calculate_precision()
        auc_pr_shuttle_kmeans = evaluator_shuttle_kmeans.calculate_auc_pr()

        print("Results: HTTP one class KMeans")
        print("Accuracy: ", accuracy_shuttle_kmeans)
        print("Recall: ", recall_shuttle_kmeans)
        print("Precision: ", precision_shuttle_kmeans)
        print("AUC PR: ", auc_pr_shuttle_kmeans)
        print("Results: HTTP one class KMeans")


    def print_metrics(self, Ytest, ypred):
        print(ypred)
        # show fraction of outliers
        print("Outliers predicted fraction:")
        print(sum(ypred) / len(ypred))
        print("Metrics")
        # compute accuracy and recall precision using sklearn metrics with printed names
        print("Accuracy")
        print(accuracy_score(Ytest, ypred))
        print("Recall")
        print(recall_score(Ytest, ypred))
        print("Precision")
        print(precision_score(Ytest, ypred))






