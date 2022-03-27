# Traffic Sign Classification

In this experiment, our aim is to classify 23 different traffic signals. Our problem uses traffic signal images of size 32x32, as the inputs to our system. Which are preprocessed by normalization and do the feasibility study of principal component analysis. Initially, we start with classifying between only 2 traffic signs using Logistic regression. Which yielded approximately 97% accuracy. and then move on to multiclass classification using Softmax regression, Which yielded approximately 90% accuracy. The dataset is split into trian, validation and testing was done using a 10-fold cross validation set, to test our accuracy and select hyperparameters. In addition, a feasibility analysis was done between aligned v/s unaligned dataset and batch v/s stochastic gradient descent.

Report can be found [here](https://drive.google.com/file/d/1fcougLmQ2WxkVt_RXXGTF5iOVsJcZyan/view?usp=sharing)



### Installation

This code uses Python 3.6.

- Install dependencies
```bash
$ pip install -r requirements.txt
```


### Execution

The following will run the program with default config.
```bash
$ python main.py 
```

The following arguments (with the default values) can be passed with the above command:
```bash
--batch-size: 1 
--epochs: 100
--learning-rate: 0.001
--k-folds: 10
--classification: multi
--gradient-descent: sgd
--align: True
--pca: True
--pca-dim: 100
```

There are few other predefined setting which can be executed by the following command:
```bash
$ python main.py --parameters-mode=predefined
```

We have following predefined hyperparameter setting to choose from:

```bash
1. Logistic Regression - Unaligned
2. Logistic Regression - Aligned
3. Logistic Regression - Dangerous Curve signs
4. Softmax Regression - Batch with PCA and Aligned Dataset
5. Softmax Regression - Batch without PCA and Aligned Dataset
6. Softmax Regression - Batch with PCA and Unaligned Dataset
7. Softmax Regression - SGD with PCA and Aligned Dataset
8. Softmax Regression - Batch v/s SGD
```