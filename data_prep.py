import numpy as np
from numpy import ndarray
from sklearn.utils import shuffle


def sample_indices(array : ndarray,n: int) -> ndarray:
    #sample without replacement
    return np.random.choice(n, int(n), replace=False)

def split_binary_dataset(X,y):

    y = np.ravel(y)
    # Create a mask for Y=1 and Y=0
    mask_Y1 = y == 1
    mask_Y0 = y == 0

    # Use the mask to filter X
    X1 = X[mask_Y1]
    X0 = X[mask_Y0]

    Y1 = y[mask_Y1]
    Y0 = y[mask_Y0]


    indices = sample_indices(X0,X1.shape[0])
    X0_sampled = X0[indices]
    X0 = np.delete(X0,indices, axis=0)
    Y0_sampled = Y0[indices]
    Y0 = np.delete(Y0,indices, axis=0)


    Y_test = np.concatenate((Y0_sampled, Y1),axis=0)
    X_test = np.concatenate((X0_sampled, X1),axis=0)
    X_train = X0
    Y_train = Y0

    X_train, Y_train = shuffle(X_train, Y_train)
    X_test, Y_test = shuffle(X_test, Y_test)

    return X_train, Y_train, X_test, Y_test





