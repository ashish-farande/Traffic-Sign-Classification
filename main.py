import argparse

from Dataset import Dataset
from network import Network

PCA_DIM = 300


def get_model(input_hyperparam):
    classification = 0 if input_hyperparam.classification == "binary" else 1
    loss = 0 if input_hyperparam.classification == "binary" else 1
    return Network(input_hyperparam, classification, loss)


def get_args():
    parser = argparse.ArgumentParser(description='CSE251B PA1')
    parser.add_argument('--parameters-mode', type=str, default="args",
                        help='The parameters can be feed in "args" or use "predefined" (default: args)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--in-dim', type=int, default=32 * 32,
                        help='number of principal components to use (default: 32 * 32)')
    parser.add_argument('--out-dim', type=int, default=43,
                        help='number of outputs (default: 43)')
    parser.add_argument('--k-folds', type=int, default=10,
                        help='number of folds for cross-validation (default: 10)')
    parser.add_argument('--classification', type=str, default="multi",
                        help='binary vs multi (default: multi)')
    parser.add_argument('--gradient-descent', type=str, default="sgd",
                        help='batch vs sgd vs both (compare) (default: sgd)')
    parser.add_argument('--align', type=bool, default=True,
                        help='Align the images to the center (default: True)')
    parser.add_argument('--pca', type=bool, default=True,
                        help='Apply PCA to the dataset (default: True)')
    parser.add_argument('--pca-dim', type=int, default=100,
                        help='Output dim of PCA (default: 100)')

    parameters = parser.parse_args()

    parameters.fold_runs = 10
    parameters.classes = None

    if parameters.parameters_mode != "args":
        print("1. Q 5_b Logistic Regression Unaligned")
        print("2. Q 5_c Logistic Regression Aligned")
        print("3. Q 5_d Logistic Regression Dangerous Curve")
        print("4. Q 6_a_i Softmax Regression with PCA and Aligned Dataset")
        print("5. Q 6_a_ii Softmax Regression without PCA and Aligned Dataset")
        print("6. Q 6_a_ii Softmax Regression with PCA and Unaligned Dataset")
        print("7. Q 6_b_i SGD with PCA and Aligned Dataset")
        print("8. Q 6_b_ii Batch v/s SGD")

        choice = int(input("Enter your choice of combination:"))

        if choice == 1:
            parameters.fold_runs = 1
            parameters.align = False
            parameters.classification = "binary"
            parameters.gradient_descent = 'batch'
            parameters.classes = [7, 8]

        elif choice == 2:
            parameters.align = True
            parameters.classification = "binary"
            parameters.gradient_descent = 'batch'
            parameters.classes = [7, 8]

        elif choice == 3:
            parameters.align = True
            parameters.classification = "binary"
            parameters.gradient_descent = 'batch'
            parameters.classes = [19, 20]

        elif choice == 4:
            parameters.align = True
            parameters.pca = True
            parameters.gradient_descent = "batch"
            parameters.classification = "multi"

        elif choice == 5:
            parameters.align = True
            parameters.pca = False
            parameters.gradient_descent = "batch"
            parameters.classification = "multi"

        elif choice == 6:
            parameters.align = False
            parameters.pca = True
            parameters.gradient_descent = "batch"
            parameters.classification = "multi"

        elif choice == 7:
            parameters.align = True
            parameters.pca = True
            parameters.gradient_descent = "sgd"
            parameters.classification = "multi"

        elif choice == 8:
            parameters.align = True
            parameters.pca = True
            parameters.gradient_descent = "both"
            parameters.classification = "multi"
            parameters.learning_rate = 0.005

        else:
            print("Option Not available")
            return None

    return parameters


if __name__ == '__main__':
    hyperparameters = get_args()

    assert hyperparameters is not None

    dataset = Dataset(aligned=hyperparameters.align, filter=hyperparameters.classes)
    mini_batch = dataset.features, dataset.labels

    hyperparameters.in_dim = hyperparameters.pca_dim if hyperparameters.pca else dataset.features.shape[1]
    hyperparameters.out_dim = dataset.labels.shape[1]

    model = get_model(input_hyperparam=hyperparameters)

    model.run_fold_set(mini_batch)
