import mysklearn.myutils as myutils
import numpy as np
import math
import copy
import mysklearn.myclassifiers as myclassifiers

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
        np.random.seed(random_state)
        #np.random.randint(10) => [0,10)
    
    if shuffle: 
        for i in range(len(X)):
            # generate a random index to swap the element at i with
            rand_index = np.random.randint(len(X)) # [0, len(X))
            X[i], X[rand_index] = X[rand_index], X[i]
            y[i], y[rand_index] = y[rand_index], y[i]

    num_instances = len(X) # 8
    if isinstance(test_size, float):
        test_size = math.ceil(num_instances * test_size) # ceil(8 * 0.33)
    split_index = num_instances - test_size # 8 - 2 = 6

    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

def kfold_cross_validation(X, n_splits=5):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes: 
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    folds = [[] for _ in range(n_splits)]
    for i, x in enumerate(X):
        folds[i % n_splits].append(x)
    
    X_train_folds = []
    X_test_folds = []
    
    for test_fold in folds:
        fold_test = []
        for sample in test_fold:
            fold_test.append(X.index(sample))
        fold_train = []
        for fold in folds:
            if fold != test_fold:
                for sample in fold:
                    fold_train.append(X.index(sample))
        
        X_train_folds.append(fold_train)
        X_test_folds.append(fold_test)

    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        n_splits(int): Number of folds.
 
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes: 
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    working_X = copy.deepcopy(X)
    for i, x in enumerate(working_X):
        working_X[i].append(y[i])
    # group by the last column which is the y value that was just appended to working_X
    header = [i for i in range(len(working_X[0]))]
    _, group_subtables = myutils.group_by(working_X, header, header[-1])

    counter = 0
    folds = [[] for _ in range(n_splits)]
    for subtable in group_subtables:
        for x in subtable:
            folds[counter % n_splits].append(x)
            counter = (counter + 1) % n_splits

    X_train_folds = []
    X_test_folds = []
    
    for test_fold in folds:
        fold_test = []
        for sample in test_fold:
            fold_test.append(working_X.index(sample))
        fold_train = []
        for fold in folds:
            if fold != test_fold:
                for sample in fold:
                    fold_train.append(working_X.index(sample))
        
        X_train_folds.append(fold_train)
        X_test_folds.append(fold_test)

    return X_train_folds, X_test_folds

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
    for index, y in enumerate(y_true):
        if y_pred[index] is not None:
            i = labels.index(y)
            j = labels.index(y_pred[index])
            matrix[i][j] += 1

    return matrix

def tune_parameters(N_range, M_range, F_range, table):
    max_percentage = 0
    max_parameters = [0, 0, 0]
    for N in range(N_range[0], N_range[1], 10):
        for M in range(M_range[0], M_range[1], 2):
            for F in range(F_range[0], F_range[1]):
                percentages = []
                for _ in range(5):
                    X = [row[:-1] for row in table]
                    y = [row[-1] for row in table]
                    X_remainder, X_test, y_remainder, y_test = train_test_split(X, y)
                    rf_classifier = myclassifiers.MyRandomForestClassifier()
                    rf_classifier.fit(X_remainder, y_remainder, N=100, M=10, F=4)
                    y_predicted = rf_classifier.predict(X_test)
                    correct = 0
                    for i, y in enumerate(y_test):
                        if y == y_predicted[i]:
                            correct += 1
                    percentage_correct = correct / len(y_test) * 100
                    percentages.append(percentage_correct)
                total_percentage = sum(percentages) / len(percentages)
                if total_percentage > max_percentage:
                    max_percentage = total_percentage
                    max_parameters = [N, M, F]
    print("N = {}, M = {}, F = {}: {}% Correct".format(max_parameters[0], max_parameters[1], max_parameters[2], int(max_percentage)))