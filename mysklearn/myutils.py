import mysklearn.myevaluation as myevaluation

import math
import copy
import random

def compute_euclidean_distance(v1, v2):
    assert len(v1) == len(v2)

    #dist = math.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
    
    vals = []
    for i in range(len(v1)):
        if isinstance(v1[i], int) or isinstance(v1[i], float):
            vals.append((v1[i] - v2[i]) ** 2)
        elif v1[i] == v2[i]:
            vals.append(0)
        else:
            vals.append(1)
    
    dist = math.sqrt(sum(vals))
    
    return dist

def sort_parallel_lists(sort_list, parallel_list):
    zipped = list(zip(sort_list, parallel_list))
    zipped.sort()
    sorted_list = [value for (value, _) in zipped]
    sorted_parallel_list = [value for (_, value) in zipped]
    
    return sorted_list, sorted_parallel_list

def get_column(table, header, col_name):
    col_index = header.index(col_name)
    col = []
    for row in table: 
        # ignore missing values ("NA" or "N/A")
        if row[col_index] != "NA" and row[col_index] != "N/A":
            col.append(row[col_index])
    return col

def group_by(table, header, group_by_col_name):
    col = get_column(table, header, group_by_col_name)
    col_index = header.index(group_by_col_name)
    
    # we need the unique values for our group by column
    group_names = sorted(list(set(col))) # e.g. 74, 75, 76, 77
    group_subtables = [[] for _ in group_names] # [[], [], [], []]
    
    # algorithm: walk through each row and assign it to the appropriate
    # subtable based on its group_by_col_name value
    for row in table:
        group_by_value = row[col_index]
        # which subtable to put this row in?
        group_index = group_names.index(group_by_value)
        group_subtables[group_index].append(row.copy()) # shallow copy
    
    return group_names, group_subtables

def print_classifier_info(test_data, y_predicted, y_actual):
    # print out data
    for i, test_instance in enumerate(test_data):
        print("instance:", ", ".join([str(val) for val in test_data[i]]))
        print("class: {}, actual: {}".format(y_predicted[i], y_actual[i]))
        
def normalize_data(x):
    values = copy.deepcopy(x)
    max_val = max(values)
    min_val = min(values)
    for val in values:
        val = (val - min_val) / ((max_val - min_val) * 1.0)
    return values

def transform_weight(value):
    if value <= 1999:
        return 1
    elif value < 2500:
        return 2
    elif value < 3000:
        return 3
    elif value < 3500:
        return 4
    else:
        return 5
    
def transform_mpg(value):
    if value <= 13:
        return 1
    elif value < 15:
        return 2
    elif value < 17:
        return 3
    elif value < 20:
        return 4
    elif value < 24:
        return 5
    elif value < 27:
        return 6
    elif value < 31:
        return 7
    elif value < 37:
        return 8
    elif value < 45:
        return 9
    else:
        return 10
    
def calculate_accuracy(matrix):
    total = 0
    true_values = 0
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if i == j:
                true_values += val
            total += val
            
    return true_values / total
            
    
def calculate_error_rate(matrix):
    return 1 - calculate_accuracy(matrix)

def predict_helper(tree, instance):
    """Recursive helper for MyDecisionTreeClassifier.predict()

    Args:
        tree (nested lists): formatted nested lists in the form created by fit()
        instance (list of obj): the current instance being predicted

    Returns:
        label (str): the class label for the current instance
    """
    info_type = tree[0]
    if info_type == "Attribute":
        attribute_index = int(tree[1][-1]) # last char in string which is the index of the attribute
        instance_value = instance[attribute_index]
        
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value:
                return predict_helper(value_list[2], instance)
    else: # leaf node
        return tree[1] # leaf class label

def get_attribute_domains(header, data):
    att_domains = {}
    for att_index, attribute in enumerate(header):
        att_values = [row[att_index] for row in data]
        att_domain = list(set(att_values))
        att_domains[attribute] = att_domain
    
    return att_domains

def calculate_entropy(attribute, instances):
    att_index = int(attribute[-1])
    possible_values = []
    value_counts = []
    for instance in instances:
        if instance[att_index] not in possible_values:
            possible_values.append(instance[att_index])
            value_counts.append(1)
        else:
            value_counts[possible_values.index(instance[att_index])] += 1            
    
    entropy = 0.0
    if sum(value_counts) > 0:
        for value_index, value in enumerate(possible_values):
            classifications = []
            class_counts = []
            for instance in instances:
                if instance[att_index] == value:
                    if instance[-1] not in classifications:
                        classifications.append(instance[-1])
                        class_counts.append(1)
                    else:
                        class_counts[classifications.index(instance[-1])] += 1
            curr_e = 0.0
            for count in class_counts:
                if (count > 0):
                    curr_val = count / value_counts[value_index]
                    curr_e += (-curr_val) * math.log(curr_val, 2)
            entropy += value_counts[value_index] / sum(value_counts) * curr_e

    return entropy

def select_attribute(instances, available_attributes):
    entropies = [calculate_entropy(attribute, instances) for attribute in available_attributes]
    min_index = entropies.index(min(entropies))

    return available_attributes[min_index]

def partition_instances(instances, split_attribute, header, attribute_domains):
    attribute_domain = attribute_domains[split_attribute]
    attribute_index = header.index(split_attribute)
    partitions = {}
    
    for attribute_value in attribute_domain:
        partitions[attribute_value] = []
        for instance in instances:
            if instance[attribute_index] == attribute_value:
                partitions[attribute_value].append(instance)

    return partitions

def all_same_class(instances):
    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False 
    return True

def tdidt_fit(current_instances, available_attributes, header, attribute_domains):
    # basic approach (uses recursion!!):

    # select an attribute to split on
    split_attribute = select_attribute(current_instances, available_attributes)
    # print("splitting on:", split_attribute)
    available_attributes.remove(split_attribute)
    # cannot split on the same attribute twice in a branch
    # recall: python is pass by object reference!!
    tree = ["Attribute", split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, header, attribute_domains)
#     print("partitions:", partitions)

    # for each partition, repeat unless one of the following occurs (base case)
    for attribute_value, partition in partitions.items():
        # print("working with partition for:", attribute_value)
        value_subtree = ["Value", attribute_value]

        # CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and all_same_class(partition):
            leaf_class = partition[0][-1]
            denominator = sum([len(part) for _, part in partitions.items()])
            leaf_node = ["Leaf", leaf_class, len(partition), denominator]
            value_subtree.append(leaf_node)

        # CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif (len(partition) > 0 and len(available_attributes) == 0):
            vals = []
            val_counts = []
            denominator = 0
            for row in partition:
                val = row[-1]
                if val not in vals:
                    vals.append(val)
                    val_counts.append(0)
                val_counts[vals.index(val)] += 1
                denominator += 1
            max_val = max(val_counts)
            max_class = vals[val_counts.index(max_val)]
            leaf_node = ["Leaf", max_class, max_val, denominator]
            value_subtree.append(leaf_node)

        # CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            return None

        else: # all base cases are false... recurse!!
            subtree = tdidt_fit(partition, available_attributes.copy(), header, attribute_domains)
            if subtree is not None:
                value_subtree.append(subtree)
            else:
                vals = []
                val_counts = []
                denominator = 0
                for row in partition:
                    val = row[-1]
                    if val not in vals:
                        vals.append(val)
                        val_counts.append(0)
                    val_counts[vals.index(val)] += 1
                    denominator += 1
                max_val = max(val_counts)
                max_class = vals[val_counts.index(max_val)]
                leaf_node = ["Leaf", max_class, max_val, denominator]
                value_subtree.append(leaf_node)
        
        tree.append(value_subtree)
    
    return tree

def print_decision_rules_helper(tree, class_name, attribute_names, curr_str, isNotRoot=True):
    if tree[0] == "Leaf":
        curr_str += "THEN {} = {}".format(class_name, tree[1])
        print(curr_str)
    elif tree[0] == "Attribute":
        for i in range(2, len(tree)): # for each value subtree
            new_str = curr_str
            if isNotRoot:
                new_str += "AND"
            new_str += " {} == {} ".format(attribute_names[int(tree[1][-1])], tree[i][1])
            print_decision_rules_helper(tree[i][2], class_name, attribute_names, new_str)
    else:
        print("ERROR")

def compute_bootstrapped_sample(table):
    """Creates a bootstrapped sample from the table

    Args:
        table (list of lists): The list of the instances to be bootstrap sampled

    Returns:
        sample (list of lists): The list of the instances in the bootstrap sample
    
    """
    n = len(table)
    sample = []
    for _ in range(n):
        rand_index = random.randrange(0, n)
        sample.append(table[rand_index])
    return sample

def compute_random_subset(values, num_values):
    shuffled = values[:] # shallow copy 
    random.shuffle(shuffled)
    return sorted(shuffled[:num_values])