import numpy as np
import matplotlib.pyplot as plt

class PCA:
    """
    This class handles all things related to PCA for PA1.

    You can add any new parameters you need to any functions. This is an 
    outline to help you get started.

    You should run PCA on both the training and validation / testing datasets 
    using the same object.

    For the visualization of the principal components, use the internal 
    parameters that are set in `fit`.
    """
    def __init__(self, num_components):
        """
        Setup the PCA object. 

        Parameters
        ----------
        num_components : int
            The number of principal components to reduce to.
        """
        self.num_components = num_components

    def fit(self, X):
        """
        Set the internal parameters of the PCA object to the data.

        Parameters
        ----------
        X : np.array
            Training data to fit internal parameters.
        """
        self.mean = np.mean(X, axis=0)
        X = X-self.mean

        C = (np.dot(np.transpose(X),X))/(X.shape[0]-1)
        eigenValues, eigenVector = np.linalg.eigh(C)

        self.eigenVector = np.fliplr(eigenVector[:,-self.num_components:])
        self.eigenValues = np.flipud(eigenValues[-self.num_components:])


        ## Need an implementation if the number of cpixels is greater than the number of image using turk and pentland Trick


    def transform(self, X):
        """
        Use the internal parameters set with `fit` to transform data.

        Make sure you are using internal parameters computed during `fit` 
        and not recomputing parameters every time!

        Parameters
        ----------
        X : np.array - size n*k
            Data to perform dimensionality reduction on

        Returns
        -------
            Transformed dataset with lower dimensionality
        """
        if self.eigenVector is None:
            self.fit(X)
        new_X = np.dot(X, self.eigenVector)
        new_X = new_X/np.sqrt(self.eigenValues)

        # image = np.multiply(X[0,:], self.eigenVector[:,0])
        # plt.imshow(image.reshape((32,32)))
        # plt.show()
        # image = np.multiply(X[0, :], self.eigenVector[:, 1])
        # plt.imshow(image.reshape((32, 32)))
        # plt.show()
        # image = np.multiply(X[0, :], self.eigenVector[:, 2])
        # plt.imshow(image.reshape((32, 32)))
        # plt.show()
        # image = np.multiply(X[0, :], self.eigenVector[:, 3])
        # plt.imshow(image.reshape((32, 32)))
        # plt.show()

        return new_X




    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
