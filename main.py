import argparse
import network
from network import Network
import data
from pca import PCA
from Dataset import Dataset


def LogisticRegerssion(hyperparameters):

    hyperparameters.in_dim = 500
    hyperparameters.epochs = 100
    hyperparameters.learning_rate == 0.000001
    hyperparameters.k_folds = 10
    dataset = Dataset(aligned=True)
    hyperparameters.out_dim = dataset.labels.shape[1]
    logistic = Network(hyperparameters, 1, 1)
    logistic.train(dataset.features, dataset.labels)


def main(hyperparameters):
        LogisticRegerssion(hyperparameters)

parser = argparse.ArgumentParser(description = 'CSE251B PA1')
parser.add_argument('--batch-size', type = int, default = 128,
        help = 'input batch size for training (default: 128)')
parser.add_argument('--epochs', type = int, default = 300,
        help = 'number of epochs to train (default: 150)')
parser.add_argument('--learning-rate', type = float, default = 0.001,
        help = 'learning rate (default: 0.001)')
parser.add_argument('--z-score', dest = 'normalization', action='store_const',
        default = data.min_max_normalize, const = data.z_score_normalize,
        help = 'use z-score normalization on the dataset, default is min-max normalization')
parser.add_argument('--in-dim', type = int, default = 32*32, 
        help = 'number of principal components to use')
parser.add_argument('--out-dim', type = int, default = 43,
        help = 'number of outputs')
parser.add_argument('--k-folds', type = int, default = 10,
        help = 'number of folds for cross-validation')

hyperparameters = parser.parse_args()
print(hyperparameters)
main(hyperparameters)
