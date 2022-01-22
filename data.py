import numpy as np
import os

from traffic_reader import load_traffic

## Its better if I create a class dataset

def traffic_sign(aligned=True):
    if aligned:
        return load_traffic('data', kind='aligned')
    return load_traffic('data', kind='unaligned')


load_data = traffic_sign

def z_score_normalize(X, u = None, sd = None):
    """
    Performs z-score normalization on X. 

    f(x) = (x - μ) / σ
        where 
            μ = mean of x
            σ = standard deviation of x

    Parameters
    ----------
    X : np.array
        The data to min-max normalize

    Returns
    -------
        Tuple:
            Transformed dataset with mean 0 and stdev 1
            Computed statistics (mean and stdev) for the dataset to undo z-scoring.
    """
    if u is None:
        u = np.mean(X, axis=1)

    if sd is None:
        sd = np.std(X, axis=1)

    X = (X-u[:, None])/sd[:, None]
    print(np.max(X))
    print(np.mean(X))
    return X

def min_max_normalize(X, _min = None, _max = None):
    """
    Performs min-max normalization on X. 

    f(x) = (x - min(x)) / (max(x) - min(x))

    Parameters
    ----------
    X : np.array
        The data to min-max normalize

    Returns
    -------
        Tuple:
            Transformed dataset with all values in [0,1]
            Computed statistics (min and max) for the dataset to undo min-max normalization.
    """
    if _min is None:
        _min = np.amin(X, axis=1)

    if _max is None:
        _max = np.amax(X, axis=1)

    X = (X-_min[:, None])/(_max-_min)[:, None]
    return X, (_min, _max)


def onehot_encode(y):
    """
    Performs one-hot encoding on y.

    Ideas:
        NumPy's `eye` function

    Parameters
    ----------
    y : np.array
        1d array (length n) of targets (k)

    Returns
    -------
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.
    """
    return np.eye(np.max(y)+1)[y]

def onehot_decode(y):
    """
    Performs one-hot decoding on y.

    Ideas:
        NumPy's `argmax` function 

    Parameters
    ----------
    y : np.array
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.

    Returns
    -------
        1d array (length n) of targets (k)
    """
    return np.argmax(y, axis=1)

def shuffle(dataset):
    """
    Shuffle dataset.

    Make sure that corresponding images and labels are kept together. 
    Ideas: 
        NumPy array indexing 
            https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing

    Parameters
    ----------
    dataset
        Tuple containing
            Images (X)
            Labels (y)

    Returns
    -------
        Tuple containing
            Images (X)
            Labels (y)
    """
    pass

def append_bias(X):
    return np.insert(X,[0,0],1)


def generate_minibatches(dataset, batch_size=64):
    X, y = dataset
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        yield X[l_idx:r_idx], y[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield X[l_idx:], y[l_idx:]


def generate_k_fold_set(dataset, k = 5):
    # Be sure to modify to include train/val/test
    X, y = dataset
    order = np.random.permutation(len(X))
    fold_width = len(X) // k
    l_idx, r_idx = 0, 2*fold_width

    for i in range(k):
        train = np.concatenate([X[order[:l_idx]], X[order[r_idx:]]]), np.concatenate([y[order[:l_idx]], y[order[r_idx:]]])
        validation = X[order[l_idx:(r_idx+l_idx)//2]], y[order[l_idx:r_idx//2]]
        if i==k-1:
            test = X[order[0:fold_width]], y[order[0:fold_width]]
        else:
            test = X[order[(r_idx+l_idx)//2:r_idx]], y[order[(r_idx+l_idx)//2:r_idx]]
        # yield train, validation
        l_idx, r_idx = l_idx + fold_width, r_idx + fold_width


def filter_dataset(dataset, labels):
    new_images = []
    new_labels = []
    for i in range(len(dataset[0])):
        if dataset[1][i] in labels:
            new_images.append(dataset[0][i])
            new_labels.append(dataset[1][i])
    return [np.array(new_images), np.array(new_labels)]




# class Data:
#     def __init__(self):
#         self.aligned_data = load_data(True)
#         self.unaligned_data = load_data(False)
#
#     def get_data_label(self, is_aligned, labels = None):
#         if is_aligned:
#             return self.get_aligned_data(labels)
#         else:
#             return self.get_unaligned_data(labels)
#
#     def get_aligned_data(self, labels):
#         if labels:
#             data = [[],[]]
#             for label in labels:
#                 if self.aligned_data[1] == label:
#                     data[0] = self.aligned_data[0]
#                     data[1] = self.aligned_data[1]
#         else:
#             data = self.aligned_data
#         return data
#
#     def get_unaligned_data(self, labels):
#         if labels:
#             data = [[],[]]
#             for label in labels:
#                 if self.unaligned_data[1] == label:
#                     data[0] = self.unaligned_data[0]
#                     data[1] = self.unaligned_data[1]
#         else:
#             data = self.unaligned_data
#         return data
#