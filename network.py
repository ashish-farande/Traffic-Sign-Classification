import math

import matplotlib.pyplot as plt
import numpy as np


from Dataset import Dataset
from pca import PCA

epsilon = 1e-10

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
    exp_array = np.exp(a-np.max(a,axis=1)[:,np.newaxis])
    return exp_array / (exp_array.sum(axis=1)[:, np.newaxis])


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
    loss = -np.mean(np.multiply(t, np.log10(y + epsilon)) + np.multiply((1 - t), np.log10(1 - y + epsilon)))
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
    loss = -np.mean(np.multiply(t, np.log(y + epsilon)))
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
        self.fig_count = 0
        self.weights = np.ones((self.hyperparameters.in_dim + 1, self.hyperparameters.out_dim))

        self.confusion = np.zeros((self.hyperparameters.out_dim, self.hyperparameters.out_dim))

        if self.hyperparameters.pca:
            self.pca = PCA(self.hyperparameters.in_dim)  ## TODO: Need to come up proper number of PCs

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

    def run_fold_set(self, minibatch):
        X, y = minibatch
        k = self.hyperparameters.k_folds

        # Creating folds
        order = np.random.permutation(len(X))
        fold_width = len(X) // k
        l_idx, r_idx = 0, 2 * fold_width
        self.weights = np.ones((self.hyperparameters.in_dim + 1, self.hyperparameters.out_dim))

        # Containers for storing the accuracy and loss for each iteration
        best_test_accuracy = np.zeros((self.hyperparameters.fold_runs, 1))

        training_loss = np.zeros((self.hyperparameters.fold_runs, self.hyperparameters.epochs))
        validation_loss = np.zeros((self.hyperparameters.fold_runs, self.hyperparameters.epochs))
        training_perf = np.zeros((self.hyperparameters.fold_runs, self.hyperparameters.epochs))
        validation_perf = np.zeros((self.hyperparameters.fold_runs, self.hyperparameters.epochs))

        # For Batch v/s SGD
        training_loss_batch = np.zeros((self.hyperparameters.fold_runs, self.hyperparameters.epochs))
        training_loss_sgd = np.zeros((self.hyperparameters.fold_runs, self.hyperparameters.epochs))

        for f in range(self.hyperparameters.fold_runs):
            print("Cross Validation fold ", f)
            # Seperate the sets out
            train = Dataset(np.concatenate([X[order[:l_idx]], X[order[r_idx:]]]),
                            np.concatenate([y[order[:l_idx]], y[order[r_idx:]]]))
            validation = Dataset(X[order[l_idx:(r_idx + l_idx) // 2]], y[order[l_idx:(r_idx + l_idx) // 2]])
            test = Dataset(X[order[0:fold_width]] if f == k - 1 else X[order[(r_idx + l_idx) // 2:r_idx]],
                           y[order[0:fold_width]] if f == k - 1 else y[order[(r_idx + l_idx) // 2:r_idx]])

            # Preprocessing
            train.z_score_normalize()
            validation.z_score_normalize()
            test.z_score_normalize()

            # Fitting PCA
            if self.hyperparameters.pca:
                self.pca.fit(train.features)
                train.features = self.pca.transform(train.features)
                validation.features = self.pca.transform(validation.features)
                test.features = self.pca.transform(test.features)

            # Appending Bias
            train.append_bias()
            validation.append_bias()
            test.append_bias()

            if self.hyperparameters.gradient_descent != "both":
                # Training
                training_loss[f], training_perf[f], validation_loss[f], validation_perf[f] = self.train(train, validation)


            else:
                # Below commented code was being used when Batch vs SGD study was being done
                self.hyperparameters.gradient_descent = 'batch'
                self.weights = np.zeros((self.hyperparameters.in_dim + 1, self.hyperparameters.out_dim))
                training_loss_batch[f], training_perf[f], validation_loss[f], validation_perf[f] = self.train(train, validation)
                self.hyperparameters.gradient_descent = 'sgd'
                self.weights = np.zeros((self.hyperparameters.in_dim + 1, self.hyperparameters.out_dim))
                training_loss_sgd[f], training_perf[f], validation_loss[f], validation_perf[f] = self.train(train, validation)
                self.hyperparameters.gradient_descent = "both"

            # Testing
            best_test_accuracy[f] = self.test(test)

            l_idx, r_idx = l_idx + fold_width, r_idx + fold_width



        print("Average Accuracy is ", np.mean(best_test_accuracy))

        if self.hyperparameters.gradient_descent != "both":
            self.plot(training_loss, validation_loss, 'Loss')
            self.plot(training_perf, validation_perf, 'Accuracy')
        else:
            self.plot(training_loss_batch, training_loss_sgd, 'Batch v/s SGD')

        # Plotting the Heatmap for confusion Matrix
        if self.hyperparameters.out_dim>1:
            self.confusion = self.confusion / self.confusion.sum(axis=1)[:, None]
            plt.imshow(self.confusion)
            plt.show()

        # Visualising weights
        self.visualize_weights()

    def train(self, train, validation):
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

        training_loss = np.zeros(self.hyperparameters.epochs)
        validation_loss = np.zeros(self.hyperparameters.epochs)
        training_accuracy = np.zeros(self.hyperparameters.epochs)
        validation_accuracy = np.zeros(self.hyperparameters.epochs)


        max_loss = -math.inf
        best_weights = self.weights

        for e in range(self.hyperparameters.epochs):
            training_loss[e], training_accuracy[e], validation_loss[e], validation_accuracy[
                e] = self.stochastic_gradient_descent(train,
                                                      validation) if self.hyperparameters.gradient_descent == 'sgd' else self.batch_gradient_descent(
                train, validation)

            # To find the best weights
            if validation_loss[e] > max_loss:
                best_weights = self.weights
                max_loss = validation_loss[e]

        self.weights = best_weights
        return training_loss, training_accuracy, validation_loss, validation_accuracy

    def test(self, test):
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

        prediction = self.forward(test.features)
        accuracy = self.get_accuracy(prediction, test.labels)

        self.update_confusion_matrix(prediction, test.labels)

        print("The accuracy of the test batch is ", accuracy)
        return accuracy

    def get_loss(self, y, t):
        if self.loss == 0:
            return binary_cross_entropy(y, t)
        else:
            return multiclass_cross_entropy(y, t)

    def get_gradients(self, X, predictions, true):
        dw = -np.dot(np.transpose(X), (true - predictions))
        return dw

    def get_accuracy(self, predictions, true):
        sample_count = predictions.shape[0]
        if self.loss == 0:
            predictions = np.array([1 if i > 0.5 else 0 for i in predictions])[:, np.newaxis]
        else:
            predictions = np.argmax(predictions, axis=1)[:, np.newaxis]
            true = np.argmax(true, axis=1)[:, np.newaxis]
        return (predictions == true).sum() / sample_count

    def batch_gradient_descent(self, train, validation):
        ## Train Set
        train_prediction = self.forward(train.features)
        train_loss = self.get_loss(train_prediction, train.labels)
        train_accuracy = self.get_accuracy(train_prediction, train.labels)

        ## Validation Set
        val_prediction = self.forward(validation.features)
        val_loss = self.get_loss(val_prediction, validation.labels)
        val_accuracy = self.get_accuracy(val_prediction, validation.labels)

        ## update the weights
        self.weights -= self.hyperparameters.learning_rate * self.get_gradients(train.features, train_prediction,
                                                                                train.labels)
        return train_loss, train_accuracy, val_loss, val_accuracy

    def stochastic_gradient_descent(self, train, validation):
        train_prediction = np.zeros_like(train.labels)
        l_idx, r_idx = 0, self.hyperparameters.batch_size
        while r_idx < (train.features.shape[0]):
            train_prediction[l_idx:r_idx] = self.forward(train.features[l_idx:r_idx])
            self.weights -= self.hyperparameters.learning_rate * self.get_gradients(train.features[l_idx:r_idx],
                                                                                    train_prediction[l_idx:r_idx],
                                                                                    train.labels[l_idx:r_idx])
            l_idx, r_idx = r_idx, r_idx + self.hyperparameters.batch_size

        train_loss = self.get_loss(train_prediction, train.labels)
        train_accuracy = self.get_accuracy(train_prediction, train.labels)

        # Validation Set
        val_prediction = self.forward(validation.features)
        val_loss = self.get_loss(val_prediction, validation.labels)
        val_accuracy = self.get_accuracy(val_prediction, validation.labels)

        return train_loss, train_accuracy, val_loss, val_accuracy

    def plot(self, train, validation, title):
        self.fig_count += 1
        sd_bar = 50 if self.hyperparameters.epochs == 300 else 10
        plt.errorbar(range(self.hyperparameters.epochs), np.mean(train, axis=0), yerr=[ np.std(train, axis=0)[i] if i%sd_bar ==0 else 0 for i in range(0,self.hyperparameters.epochs)], color='y', label='Train')
        plt.errorbar(range(self.hyperparameters.epochs), np.mean(validation, axis=0), yerr=[ np.std(validation, axis=0)[i] if i%sd_bar ==0 else 0 for i in range(0,self.hyperparameters.epochs)], color='g', label='Validation')
        plt.title(title +" average for LR: "+ str(self.hyperparameters.learning_rate))
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(title)
        # plt.savefig('Fig_' + str(self.fig_count) + '.png')
        plt.show()

    def update_confusion_matrix(self, predictions, true):
        if self.hyperparameters.out_dim>1:
            predictions = np.argmax(predictions, axis=1)[:, np.newaxis]
            true = np.argmax(true, axis=1)[:, np.newaxis]
            for i in range(predictions.shape[0]):
                self.confusion[true[i]-1,predictions[i]-1] += 1
                ## As we need the average the matrix is plotted at the end


    def visualize_weights(self):
        if self.weights.shape[0] == 1025:
            labels = [7,8,33,34]
            for label in labels:
                weight = np.interp(self.weights[1:,label], (self.weights[1:,label].min(), self.weights[1:,label].max()), (0,255))
                plt.title("Class "+str(label))
                plt.imshow(weight.reshape((32,32)))
                plt.show()

