import argparse

from Dataset import Dataset
from network import Network

PCA_DIM = 300


def softmax_regression(hyperparameters, gd, is_aligned, isPCA):
    dataset = Dataset(aligned=is_aligned)
    minibatch = dataset.features, dataset.labels

    hyperparameters.epochs = 50
    hyperparameters.do_PCA = isPCA
    hyperparameters.fold_runs = 1
    hyperparameters.out_dim = dataset.labels.shape[1]
    hyperparameters.in_dim = PCA_DIM if isPCA else dataset.features.shape[1]
    hyperparameters.learning_rate = 0.001 if gd != 'both' else 0.005
    hyperparameters.gd = gd

    print(hyperparameters)

    softmax_sgd= Network(hyperparameters, 1, 1)
    softmax_sgd.run_fold_set(minibatch)



def LogisticRegerssion(hyperparameters, foldruns, is_aligned, classes):
    learning = [0.001]

    dataset = Dataset(aligned=is_aligned, filter=classes)
    minibatch = dataset.features, dataset.labels

    hyperparameters.in_dim = 100
    hyperparameters.epochs = 50
    hyperparameters.fold_runs = foldruns
    hyperparameters.gd = 'batch'
    hyperparameters.out_dim = dataset.labels.shape[1]
    print(hyperparameters)

    logistic = Network(hyperparameters, 0, 0)
    for rate in learning:
        logistic.hyperparameters.learning_rate = rate
        logistic.run_fold_set(minibatch)


def main(hyperparameters):
    hyperparameters.learning_rate = 0.01
    hyperparameters.k_folds = 10
    hyperparameters.fold_runs = 10
    hyperparameters.do_PCA = True
    hyperparameters.batch_size = 1


    print("1. Q 5_b Logistic Regression Unaligned")
    print("2. Q 5_c Logistic Regression Aligned")
    print("3. Q 5_d Logistic Regression Dangerous Curve")
    print("4. Q 6_a_i Softmax Regression with PCA and Aligned Dataset")
    print("5. Q 6_a_ii Softmax Regression without PCA and Aligned Dataset")
    print("6. Q 6_a_ii Softmax Regression with PCA and Unaligned Dataset")
    print("7. Q 6_b_i SGD with PCA and Aligned Dataset")
    print("8. Q 6_b_ii Batch v/s SGD")



    choice = int(input("Enter your choice of question:"))

    if choice==1:
        LogisticRegerssion(hyperparameters, 1, False, [7,8])
    elif choice==2:
        LogisticRegerssion(hyperparameters, 10, True, [7,8])
    elif choice == 3:
        LogisticRegerssion(hyperparameters, 10, True, [19,20])

    elif choice == 4:
        softmax_regression(hyperparameters, 'batch', True, True)
    elif choice == 5:
        softmax_regression(hyperparameters, 'batch', True, False)
    elif choice == 6:
        softmax_regression(hyperparameters, 'batch', False, True)
    elif choice == 7:
        softmax_regression(hyperparameters, 'sgd', True, True)
    elif choice == 8:
        softmax_regression(hyperparameters, 'both', True, True)
    else:
        print("Option Not available")



parser = argparse.ArgumentParser(description='CSE251B PA1')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train (default: 150)')
parser.add_argument('--learning-rate', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--in-dim', type=int, default=32 * 32,
                    help='number of principal components to use')
parser.add_argument('--out-dim', type=int, default=43,
                    help='number of outputs')
parser.add_argument('--k-folds', type=int, default=10,
                    help='number of folds for cross-validation')

hyperparameters = parser.parse_args()
main(hyperparameters)
