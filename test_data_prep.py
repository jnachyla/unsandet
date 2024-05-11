from unittest import TestCase
from data_prep import split_binary_dataset
import numpy as np

class Test(TestCase):
    def test_split_binary_dataset(self):

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])
        y = np.array([1, 0, 1, 0, 0 , 0, 0, 0 ])
        Xtrain,Ytrain,Xtest,Ytest = split_binary_dataset(X, y)
        self.assertEqual(Xtrain.shape[0], 4)
        self.assertEqual(Ytrain.shape[0], 4)
        self.assertEqual(Xtest.shape[1], 2)



