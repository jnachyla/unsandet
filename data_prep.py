import numpy as np
from numpy import ndarray
from sklearn.utils import shuffle

def sample_indices(n: int) -> ndarray:
    '''
    Sample n indices without replacement.
    :param n: number of indices to sample
    :return: array of n indices
    '''
    return np.random.choice(n, int(n), replace=False)

def split_binary_dataset(X,y, inliers_to_outliers_ratio=1.0):
    '''
    Split a binary dataset into training and testing sets.
    The train set has only inliers, while the test set has both inliers and outliers.
    The number of inliers in the test set is inliers_to_outliers_ratio times the number of outliers.
    :param X: numpy array of features
    :param y: numpy array of labels
    :param inliers_to_outliers_ratio:  The ratio of inliers to outliers in the test set.
    :return: tuple of numpy arrays (X_train, Y_train, X_test, Y_test)
    '''

    y = np.ravel(y)
    # Create a mask for Y=1 and Y=0
    mask_Y1 = y == 1
    mask_Y0 = y == 0

    # Use the mask to filter X
    X1 = X[mask_Y1]
    X0 = X[mask_Y0]

    Y1 = y[mask_Y1]
    Y0 = y[mask_Y0]

    outliers_size = X1.shape[0]
    inliers_indices = sample_indices(int(outliers_size * inliers_to_outliers_ratio))

    X0_sampled = X0[inliers_indices]
    X0 = np.delete(X0,inliers_indices, axis=0)
    Y0_sampled = Y0[inliers_indices]
    Y0 = np.delete(Y0,inliers_indices, axis=0)


    Y_test = np.concatenate((Y0_sampled, Y1),axis=0)
    X_test = np.concatenate((X0_sampled, X1),axis=0)
    X_train = X0
    Y_train = Y0

    X_train, Y_train = shuffle(X_train, Y_train)
    X_test, Y_test = shuffle(X_test, Y_test)

    return X_train, Y_train, X_test, Y_test





