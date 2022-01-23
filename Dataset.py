import numpy as np
import pickle

FILE_NAME = "./data/train_wb_"
FILE_EXT = '.p'


class Dataset:
    def __init__(self, features=None, labels=None, aligned=True, filter= None):
        if features is None and labels is None:
            self.kind = 'aligned' if aligned else 'unaligned'
            self.filter = filter
            self.load_data()
        else:
            self.features = features
            self.labels = labels

    def load_data(self):
        """Load traffic data from `path`"""
        file_name = FILE_NAME + self.kind + FILE_EXT
        with open(file_name, mode='rb') as f:
            train = pickle.load(f)
        if self.filter:
            images = []
            labels = []
            for i in range(train['labels'].shape[0]):
                if train['labels'][i] in self.filter:
                    images.append(train['features'][i])
                    labels.append(train['labels'][i])
            images = np.array(images)
            labels = np.array(labels)
        else:
            images, labels = train['features'], train['labels']

        ## TODO: Should the bias be done after or before normalization
        self.features = images.reshape((images.shape[0], -1))
        self.labels = self.onehot_encode(labels)

    def onehot_encode(self, y):
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
        return (y - np.min(y))[:,np.newaxis] if self.filter else np.eye(np.max(y) + 1)[y]

    def onehot_decode(self, y):
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
        return y+np.max(y) if self.filter else np.argmax(y, axis=1)

    def z_score_normalize(self, u=None, sd=None):
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
            u = np.mean(self.features, axis=1)

        if sd is None:
            sd = np.std(self.features, axis=1)

        self.features = (self.features - u[:, None]) / sd[:, None]
        return self.features

    def min_max_normalize(self, _min=None, _max=None):
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
        self.min = np.amin(self.features, axis=1)
        self.max = np.amax(self.features, axis=1)

        self.features = (self.features - self.min[:, None]) / (self.max - self.min)[:, None]
        return self.features

    def append_bias(self):
         self.features = np.insert(self.features, 0, 1, axis=1)

    ## Need to figure out a way
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