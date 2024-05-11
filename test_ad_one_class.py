from unittest import TestCase

import numpy as np

from ad_one_class import OneClassAnnomalyDetector
from data_prep import split_binary_dataset
import numpy as np
from sklearn.datasets import make_blobs


class TestOneClassAnnomalyDetector(TestCase):

    def create_dataset(self):
        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate inlier data
        centers = [[0, 0], [5, 5]]  # Centers of the clusters for inliers
        cluster_std = [0.5, 0.5]  # Standard deviation of the clusters
        X_inliers, _ = make_blobs(n_samples=1000, centers=centers, cluster_std=cluster_std)

        # Generate outlier data
        outlier_center = [2.5, 2.5]  # Not necessarily the center but spread range
        X_outliers = np.random.uniform(low=-2, high=8, size=(100, 2))

        # Combine the datasets
        X = np.vstack([X_inliers, X_outliers])
        y = np.hstack([np.zeros(len(X_inliers)), np.ones(len(X_outliers))])  # 0 for inliers, 1 for outliers
        return X, y
    def test_svm(self):
        X, y = self.create_dataset()

        Xtrain, Ytrain, Xtest, Ytest = split_binary_dataset(X, y)

        # test svm
        model  = OneClassAnnomalyDetector(model_name = "oneclasssvm")
        model.fit(Xtrain)

        #assert no errors
        ypred = model.predict(Xtest)
        print(ypred)



    def test_isolation_forest(self):
        X, y = self.create_dataset()

        Xtrain, Ytrain, Xtest, Ytest = split_binary_dataset(X, y)

        # test isolation forest
        model  = OneClassAnnomalyDetector(model_name = "isolationforest")
        model.fit(Xtrain)

        #assert no errors
        ypred = model.predict(Xtest)
        print(ypred)

