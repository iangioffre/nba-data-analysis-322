import mysklearn.myutils as myutils
import mysklearn.myevaluation as myevaluation
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

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        N: number of learners in our initial ensemble
        M: number of "better" learners to use from the N learners
        F: number of attributes to make available at each node
        trees(list of nested list): The list of the M best decision tree classifiers

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyRandomForestClassfier.

        """
        self.X_remainder = None 
        self.y_remainder = None
        self.N = None
        self.M = None
        self.F = None
        self.trees = None

    def fit(self, X_train, y_train, N=20, M=7, F=2):
        """Fits a random forest classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm over several trees.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
            N: number of learners in our initial ensemble
            M: number of "better" learners to use from the N learners
            F: number of attributes to make available at each node
        """
        self.X_remainder = X_train
        self.y_remainder = y_train
        self.N = N
        self.M = M
        self.F = F

        # append y to X
        data = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        
        # get bootstrapped sample of training set and train a decision tree, append to list of trees (not self.trees)
        tree_accuracies = []
        candidate_trees = []
        for _ in range(N):
            sample_train, sample_test = myutils.compute_bootstrapped_sample(data)
            sample_train_X = [row[:-1] for row in sample_train]
            sample_test_X = [row[:-1] for row in sample_test]
            sample_test_y = [row[-1] for row in sample_test]
            header = ["att{}".format(i) for i in range(len(sample_train_X[0]))]
            attribute_domains = myutils.get_attribute_domains(header, sample_train)
            available_attributes = header.copy()
            tree = myutils.tdidt_fit_rf(sample_train, available_attributes, header, attribute_domains, F)
            
            # test the decision tree on the validation set and sort by accuracy
            y_predicted = []
            for instance in sample_test_X:
                y_predicted.append(myutils.predict_helper(tree, instance))

            possible_classes = list(set(y_train))
            matrix = myevaluation.confusion_matrix(sample_test_y, y_predicted, possible_classes)
            accuracy = myutils.calculate_accuracy(matrix)

            if len(candidate_trees) < M:
                tree_accuracies.append(accuracy)
                candidate_trees.append(tree)
            elif accuracy > tree_accuracies[0]:
                tree_accuracies[0] = accuracy
                candidate_trees[0] = tree
            
            myutils.sort_parallel_lists(tree_accuracies, candidate_trees)
            
        self.trees = candidate_trees
        
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
            candidate_predictions = []
            for tree in self.trees:
                candidate_predictions.append(myutils.predict_helper(tree, instance))
            y_predicted.append(myutils.compute_majority_vote_prediction(candidate_predictions))

        return y_predicted