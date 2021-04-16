import mysklearn.myutils as myutils
import copy
import operator

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope 
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        x_train = []
        for X in X_train:
            x_train.append(X[0])
        mean_x = sum(x_train) / len(x_train)
        mean_y = sum(y_train) / len(y_train)

        m = sum([(x_train[i] - mean_x) * (y_train[i] - mean_y) for i in range(len(x_train))]) / sum([(x_train[i] - mean_x) ** 2 for i in range(len(x_train))])
        # y = mx + b => b = y - mx
        b = mean_y - m * mean_x
        self.slope = m
        self.intercept = b

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        x_test = []
        for X in X_test:
            x_test.append(X[0])
        
        y_predicted = []
        for x in x_test:
            y = self.slope * x + self.intercept
            y_predicted.append(y)
        
        return y_predicted

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

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.priors = None 
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        # store priors as a dict (dict[class] = probability)
        # store posteriors as a dict of dicts of dicts (dict[class][attr][value] = probability)

        self.X_train = X_train
        self.y_train = y_train

        # find possible classifications
        classifications = list(set(y_train))
        class_counts = {}

        # calculate priors
        for classification in classifications:
            class_counts[classification] = 0

        for y in y_train:
            class_counts[y] += 1

        self.priors = {}
        for classification in classifications:
            self.priors[classification] = class_counts[classification] / len(y_train)

        # find possible values of each attribute
        # possible_values[i]: list of possible values for attribute i
        num_attributes = len(X_train[0])
        possible_values = [[] for _ in range(num_attributes)]
        for row in X_train:
            for i, val in enumerate(row):
                if val not in possible_values[i]:
                    possible_values[i].append(val)

        # calculate posteriors
        posterior_counts = {}
        for classification in classifications:
            posterior_counts[classification] = {}
            for i in range(num_attributes):
                posterior_counts[classification][i] = {}

        for classification in classifications:
            for i in range(num_attributes):
                for val in possible_values[i]:
                    posterior_counts[classification][i][val] = 0

        for i, row in enumerate(X_train):
            for j, val in enumerate(row):
                posterior_counts[y_train[i]][j][val] += 1


        self.posteriors = {}
        for classification in classifications:
            self.posteriors[classification] = {}
            for i in range(num_attributes):
                self.posteriors[classification][i] = {}

        for classification in classifications:
            for i in range(num_attributes):
                for val in possible_values[i]:
                    self.posteriors[classification][i][val] = posterior_counts[classification][i][val] / class_counts[classification]

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        classifications = list(self.priors.keys())
        y_predicted = []
        for row in X_test:
            prediction_values = []
            for classification in classifications:
                value = self.priors[classification]
                for i, val in enumerate(row):
                    value *= self.posteriors[classification][i][val]
                prediction_values.append(value)
            max_value = max(prediction_values)
            prediction = classifications[prediction_values.index(max_value)]
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

    # BONUS METHOD
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).

        Notes: 
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this
