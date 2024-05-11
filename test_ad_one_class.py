from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score

from ad_one_class import OneClassAnnomalyDetector
from data_prep import split_binary_dataset
import numpy as np
from sklearn.datasets import make_blobs


class TestOneClassAnnomalyDetector(TestCase):

    def create_dataset(self):
        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate inlier data
        centers = [[10.0, 10.0, 10.0]]
        cluster_std = [0.5]
        X_inliers, _ = make_blobs(n_features=3,n_samples=1000, centers=centers, cluster_std=cluster_std)

        # Generate outlier data
        outlier_center = [[2.5, 2.5, 2.5]]
        X_outliers, _ = make_blobs(n_features=3,n_samples=100, centers=outlier_center, cluster_std=cluster_std)

        # Combine the datasets
        X = np.vstack([X_inliers, X_outliers])
        y = np.hstack([np.zeros(len(X_inliers)), np.ones(len(X_outliers))])  # 0 for inliers, 1 for outliers

        # Visualize the data inliers and outliers which has features
        plt.scatter(X_inliers[:, 0], X_inliers[:, 1], label="inliers")
        plt.scatter(X_outliers[:, 0], X_outliers[:, 1], label="outliers")
        plt.legend()
        plt.show()

        return X, y
    def test_svm(self):
        X, y = self.create_dataset()

        Xtrain, Ytrain, Xtest, Ytest = split_binary_dataset(X, y)

        # test svm
        model  = OneClassAnnomalyDetector(model_name = "oneclasssvm")
        model.fit(Xtrain)

        #assert no errors
        ypred = model.predict(Xtest)

        self.print_metrics(Ytest, ypred)



    def test_isolation_forest(self):
        X, y = self.create_dataset()

        Xtrain, Ytrain, Xtest, Ytest = split_binary_dataset(X, y)

        # test isolation forest
        model  = OneClassAnnomalyDetector(model_name = "isolationforest")
        model.fit(Xtrain)

        #assert no errors
        ypred = model.predict(Xtest)
        self.print_metrics(Ytest, ypred)

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



