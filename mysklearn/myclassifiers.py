import mysklearn.myutils as myutils
import copy
import operator

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train 
        self.y_train = y_train 

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        for x_test in X_test:
            x_train = copy.deepcopy(self.X_train)
            for i, instance in enumerate(x_train):
                # append the class label
                instance.append(self.y_train[i])
                # append the original row index
                instance.append(i)
                # append the distance to [2, 3]
                dist = myutils.compute_euclidean_distance(instance[:len(x_test)], x_test)
                instance.append(dist)
            # [...data, classification, original index, distance]

            train_sorted = sorted(x_train, key=operator.itemgetter(-1))
            top_k = train_sorted[:self.n_neighbors]

            instance_distances = []
            instance_neighbor_indices = []
            for row in top_k:
                instance_distances.append(row[-1])
                instance_neighbor_indices.append(row[-2])
            distances.append(instance_distances)
            neighbor_indices.append(instance_neighbor_indices)
        
        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        _, neighbor_indices = self.kneighbors(X_test)
        for indices in neighbor_indices:
            classifications = []
            counts = []
            for index in indices:
                classification = self.y_train[index]
                if classification not in classifications:
                    classifications.append(classification)
                    counts.append(1)
                else:
                    counts[classifications.index(classification)] += 1
            prediction = classifications[counts.index(max(counts))]
            y_predicted.append(prediction)
        
        return y_predicted

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        header = ["att{}".format(i) for i in range(len(X_train[0]))]
        attribute_domains = myutils.get_attribute_domains(header, X_train)
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        available_attributes = header.copy()
        self.tree = myutils.tdidt_fit(train, available_attributes, header, attribute_domains)
        # TODO: (for random forest) pass a subset of the available_attributes (in tdidt_fit()) at each node
        #       using myutils.compute_random_subset(header, F) where F > 2
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        for instance in X_test:
            y_predicted.append(myutils.predict_helper(self.tree, instance))

        return y_predicted

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        if attribute_names is None:
            attribute_names = ["att{}".format(i) for i in range(len(self.X_train[0]))]

        myutils.print_decision_rules_helper(self.tree, class_name, attribute_names, "IF", isNotRoot=False)

class MyRandomForestClassfier:
    """Represents a Random Forest classifier

    TODO: update docstring

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        trees(list of nested list): The list of the M best decision tree classifiers

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyRandomForestClassfier.

        """
        # TODO: add necessary attributes
        self.X_train = None 
        self.y_train = None
        self.trees = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        TODO: update docstring

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        # split data into training set and validation set
        # get bootstrapped sample of training set and train a decision tree, append to list of trees (not self.trees)
        # test the decision trees on the validation set and sort by accuracy
        # take the M best decision trees and put into self.trees (self.trees = trees[0:M] where trees is sorted best to worst accuracy)

        # a more RAM sensitive approach:
        # keep len(trees) at M best trees and check each time if the accuracy of the new tree is better than the least accurate tree in trees
        # if better, replace the least accurate tree with the new tree
        # (to do this, sorting of parallel lists is necessary - myutils.sort_parallel_lists(accuracies, trees))

        pass # TODO: implement fit
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        TODO: update docstring

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # TODO: implement predict