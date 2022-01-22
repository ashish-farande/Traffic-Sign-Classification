import math

import matplotlib.pyplot as plt
import numpy as np

from Dataset import Dataset
from pca import PCA

# import tqdm
epsilon = 1e-5
"""
NOTE
----
Start by implementing your methods in non-vectorized format - use loops and other basic programming constructs.
Once you're sure everything works, use NumPy's vector operations (dot products, etc.) to speed up your network.
"""


def sigmoid(a):
    """
    Compute the sigmoid function.

    f(x) = 1 / (1 + e ^ (-x))

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying sigmoid (z from the slides).
    """
    return 1.0 / (1 + np.exp(-a))


def softmax(a):
    """
    Compute the softmax function.

    f(x) = (e^x) / Σ (e^x)

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying softmax (z from the slides).
    """
    exp_array = np.exp(a - np.max(a))
    return exp_array / exp_array.sum()


def binary_cross_entropy(y, t):
    """
    Compute binary cross entropy.

    L(x) = t*ln(y) + (1-t)*ln(1-y)

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        binary cross entropy loss value according to above definition
    """
    t = (t[:, None])
    loss = np.mean(np.multiply(t, np.log10(y + epsilon)) + np.multiply((1 - t), np.log10(1 - y + epsilon)))
    return loss


def multiclass_cross_entropy(y, t):
    """
    Compute multiclass cross entropy.

    L(x) = - Σ (t*ln(y))

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        multiclass cross entropy loss value according to above definition
    """
    loss = -np.multiply(t, np.log(y + epsilon)).sum()
    return loss


class Network:
    def __init__(self, hyperparameters, activation, loss):
        """
        Perform required setup for the network.

        Initialize the weight matrix, set the activation function, save hyperparameters.

        You may want to create arrays to save the loss values during training.

        Parameters
        ----------
        hyperparameters
            A Namespace object from `argparse` containing the hyperparameters
        activation
            The non-linear activation function to use for the network
        loss
            The loss function to use while training and testing
        """
        self.hyperparameters = hyperparameters
        self.activation = activation
        self.loss = loss

        self.weights = np.ones((hyperparameters.in_dim + 1, hyperparameters.out_dim))

    def forward(self, X):
        """
        Apply the model to the given patterns

        Use `self.weights` and `self.activation` to compute the network's output

        f(x) = σ(w*x)
            where
                σ = non-linear activation function
                w = weight matrix

        Make sure you are using matrix multiplication when you vectorize your code!

        Parameters
        ----------
        X
            Patterns to create outputs for
        """
        if self.activation == 0:
            y = sigmoid(np.dot(X, self.weights))
        else:
            y = softmax(np.dot(X, self.weights))

        return y

    def __call__(self, X):
        return self.forward(X)

    def train(self, feature, label):
        """
        Train the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` and the gradient defined in the slides to update the network.

        Parameters
        ----------
        minibatch
            The minibatch to iterate over

        Returns
        -------
        Tuple
            average loss over the minibatch
            accuracy over the minibatch
        """
        print("Started Training")
        X, y = feature, label
        k = self.hyperparameters.k_folds

        ## Creating folds
        order = np.random.permutation(len(X))
        fold_width = len(X) // k
        l_idx, r_idx = 0, 2 * fold_width

        best_test_loss = []
        best_test_accuracy = []

        training_loss = np.zeros((k, self.hyperparameters.epochs))
        validation_loss = np.zeros((k, self.hyperparameters.epochs))
        training_perf = np.zeros((k, self.hyperparameters.epochs))
        validation_perf = np.zeros((k, self.hyperparameters.epochs))

        for f in range(k):
            print("Cross Validation fold ", f)
            train = Dataset(np.concatenate([X[order[:l_idx]], X[order[r_idx:]]]),
                            np.concatenate([y[order[:l_idx]], y[order[r_idx:]]]))
            validation = Dataset(X[order[l_idx:(r_idx + l_idx) // 2]], y[order[l_idx:(r_idx + l_idx) // 2]])
            test = Dataset(X[order[0:fold_width]] if f == k - 1 else X[order[(r_idx + l_idx) // 2:r_idx]],
                           y[order[0:fold_width]] if f == k - 1 else y[order[(r_idx + l_idx) // 2:r_idx]])
            l_idx, r_idx = l_idx + fold_width, r_idx + fold_width

            train.z_score_normalize()
            validation.z_score_normalize()
            test.z_score_normalize()

            ## Fitting PCA
            pca = PCA(self.hyperparameters.in_dim)  ## TODO: Need to come up proper number of PCs
            pca.fit(train.features)
            train.features = pca.transform(train.features)
            validation.features = pca.transform(validation.features)
            test.features = pca.transform(test.features)

            train.append_bias()
            validation.append_bias()
            test.append_bias()

            min_loss = math.inf
            # self.weights = np.ones((self.hyperparameters.in_dim + 1, self.hyperparameters.out_dim))
            best_weights = self.weights

            validation_accuracy = []

            test_losses = []
            test_accuracy = []
            for e in range(self.hyperparameters.epochs):
                train_pred = self.forward(train.features)
                training_loss[f, e] = self.getLoss(train_pred, train.labels)

                training_perf[f, e] = self.getAccuracy(train_pred, train.labels)

                ## For plotting
                val_pred = self.forward(validation.features)
                validation_loss[f, e] = self.getLoss(val_pred, validation.labels)
                validation_perf[f, e] = self.getAccuracy(val_pred, validation.labels)
                test_pred = self.forward(test.features)
                test_losses.append(self.getLoss(test_pred, test.labels))
                test_accuracy.append(self.getAccuracy(test_pred, test.labels))

                ## To find the best weights
                if validation_loss[f][e] < min_loss:
                    best_weights = self.weights

                ## update the weights
                self.weights -= self.hyperparameters.learning_rate * self.getGradients(train.features, train_pred,
                                                                                       train.labels)

            self.weights = best_weights
            best_test_pred = self.forward(test.features)
            best_test_loss.append(self.getLoss(best_test_pred, test.labels))
            best_test_accuracy.append(self.getAccuracy(best_test_pred, test.labels))

        print("Final Accuracy is ", np.mean(best_test_accuracy))
        plt.plot(range(self.hyperparameters.epochs), np.mean(training_loss, axis=0))
        plt.title('Training Set Loss')
        plt.show()

        plt.plot(range(self.hyperparameters.epochs), np.mean(validation_loss, axis=0))
        plt.title('Validation Set Loss')
        plt.show()

        plt.plot(range(self.hyperparameters.epochs), np.mean(training_perf, axis=0))
        plt.title('Training Set Accuracy')
        plt.show()

        plt.plot(range(self.hyperparameters.epochs), np.mean(validation_perf, axis=0))
        plt.title('Validation Set Accuracy')
        plt.show()

    def test(self, minibatch):
        """
        Test the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` to compute the loss.
        Do NOT update the weights in this method!

        Parameters
        ----------
        minibatch
            The minibatch to iterate over
        """
        X, y = minibatch

        pass

    def getLoss(self, y, t):
        if self.loss == 0:
            return binary_cross_entropy(y, t)
        else:
            return multiclass_cross_entropy(y, t)

    def getGradients(self, X, predictions, true):
        if self.loss == 0:
            true = true[:, np.newaxis]
            predictions = np.array([1 if i > 0.5 else 0 for i in predictions])[:, np.newaxis]
        dw = -np.dot(np.transpose(X), (true - predictions))
        return dw

    def getAccuracy(self, predictions, true):
        sample_count = predictions.shape[0]
        if self.loss == 0:
            predictions = np.array([1 if i > 0.5 else 0 for i in predictions])[:, np.newaxis]
            true = true[:, np.newaxis]
        else:
            predictions = np.argmax(predictions, axis=1)[:, np.newaxis]
            true = np.argmax(true, axis=1)[:, np.newaxis]
        return (predictions == true).sum() / sample_count

    def batch_gradient_descent(self, train, validation, test):

        ## Train Set
        train_prediction = self.forward(train.features)
        train_loss = self.getLoss(train_prediction, train.labels)
        train_accuracy = self.getAccuracy(train_prediction, train.labels)

        ## Validation Set
        val_prediction = self.forward(validation.features)
        val_loss = self.getLoss(val_prediction, validation.labels)
        val_accuracy = self.getAccuracy(val_prediction, validation.labels)

        ## update the weights
        self.weights -= self.hyperparameters.learning_rate * self.getGradients(train.features, train_prediction,
                                                                               train.labels)
