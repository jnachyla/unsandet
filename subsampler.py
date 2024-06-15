import pandas as pd
import scipy
import numpy as np


def subsample_majority_class(X, y, fraction=0.5):
    """
    Subsample the majority class in a binary classification dataset.

    Parameters:
    X : np.ndarray
        The input features array.
    y : np.ndarray
        The labels array.
    fraction : float, optional (default=0.5)
        The fraction by which to subsample the majority class.

    Returns:
    X_resampled : np.ndarray
        The resampled input features array.
    y_resampled : np.ndarray
        The resampled labels array.


    Example usage:
    X = np.array([...])  # Feature array
    y = np.array([...])  # Label array
    X_resampled, y_resampled = subsample_majority_class(X, y, fraction=0.5)
    """
    # Identify the majority and minority classes
    unique, counts = np.unique(y, return_counts=True)
    majority_class = unique[np.argmax(counts)]
    minority_class = unique[np.argmin(counts)]

    # Indices of the majority and minority classes
    majority_indices = np.where(y == majority_class)[0]
    minority_indices = np.where(y == minority_class)[0]

    # Number of samples to keep from the majority class
    n_majority_keep = int(len(majority_indices) * fraction)

    # Randomly subsample the majority class
    np.random.seed(42)  # For reproducibility
    majority_indices_subsampled = np.random.choice(majority_indices, n_majority_keep, replace=False)

    # Combine the minority class indices and subsampled majority class indices
    resampled_indices = np.concatenate([minority_indices, majority_indices_subsampled])

    # Shuffle the combined indices to mix the classes
    np.random.shuffle(resampled_indices)

    # Create the resampled dataset
    X_resampled = X[resampled_indices]
    y_resampled = y[resampled_indices]

    return X_resampled, y_resampled




def test():

    shuttle = scipy.io.loadmat("shuttle.mat")
    shuttle_data = pd.DataFrame(shuttle['X'])
    shuttle_eval = pd.DataFrame(shuttle['y'])

    subsampled = subsample_majority_class(shuttle_data.values, shuttle_eval.values, fraction=0.5)
    assert subsampled[0].shape[0] < shuttle_data.shape[0]* 0.6
    assert subsampled[0].shape[0] > shuttle_data.shape[0]* 0.5
    assert subsampled[0].shape[0] == subsampled[1].shape[0]


def test_subsample_majority_class():
    # Test data
    X = np.array([
        [1, 2], [2, 3], [3, 4],  # Class 0
        [4, 5], [5, 6], [6, 7], [7, 8],  # Class 1
        [8, 9], [9, 10], [10, 11], [11, 12], [12, 13]
    ])
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])  # Imbalanced with 3 class 0 and 7 class 1

    # Expected results
    expected_X_shape = (6,)  # 3 minority class + 4 (7*0.5) majority class
    expected_y_counts = {0: 3, 1: 3}  # 3 of class 0 and 4 of class 1

    # Perform subsampling
    X_resampled, y_resampled = subsample_majority_class(X, y, fraction=0.5)

    # Check the shape of the resampled dataset
    assert X_resampled.shape[0] == expected_X_shape[0], f"Expected {expected_X_shape[0]} samples, got {X_resampled.shape[0]} samples."

    # Check the distribution of the classes in the resampled dataset
    unique, counts = np.unique(y_resampled, return_counts=True)
    resampled_counts = dict(zip(unique, counts))
    assert resampled_counts == expected_y_counts, f"Expected class counts {expected_y_counts}, got {resampled_counts}."



