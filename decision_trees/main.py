import pandas as pd
import numpy as np
import math


'''
    Training section
    Part 1: Model creation
'''

input_df = pd.read_csv("data_banknote_authentication.csv", header=None)

# dataset = []
# sensorlessDriveData = pd.read_csv("Sensorless_drive_diagnosis.csv", delim_whitespace=True, header=None)
# for df_row in sensorlessDriveData.values:
#     df_row[-1] -= 1
#     dataset.append(df_row)
# dataset = np.array(dataset)
# input_df = pd.DataFrame(dataset)

dataset = input_df.values
test_data = []
rowCount = 1
for bagRow in dataset:
    if rowCount % 10 == 0:
        test_data.append(bagRow)
    rowCount += 1
test_data = np.array(test_data)
rowCount = 1
train_data = []
for bagRow in dataset:
    if rowCount % 10 != 0:
        train_data.append(bagRow)
    rowCount += 1
train_data = np.array(train_data)
np.random.shuffle(train_data)
max_depth = 11


########## Initializing Tree
class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None
        self.split_col = None
        self.split_val = None
        self.dominant = None
        self.labelCount = None


def tree_creator(tree_data):
    root = Tree()
    root.data = tree_data
    return root


def split_data(data, f_index, value):  # Splitting the dataset
    left = []
    right = []
    for row in data:

        if row[f_index] > value:
            right.append(row)
        else:
            left.append(row)
    return left, right


def split_entropy(groups, classes):  # Finding total entropy of splits
    entropy = 0

    for group in groups:

        normalized_group_size = len(group) / len(train_data)
        group_sum = 0
        label_counts = [0] * len(classes)

        for row in group:
            label_counts[int(row[-1])] += 1

        for label in classes:

            try:
                group_sum -= (label_counts[int(label)] / len(group)) * math.log2(
                    (label_counts[int(label)] / len(group)))
            except ValueError:
                group_sum -= 0
        entropy += normalized_group_size * group_sum

    return entropy


def get_best_split(data, cur_node, depth):  # Finding best splits
    entropy = 0
    classes = np.unique(dataset[:, -1])
    label_counts = [0] * len(classes)
    for row in data:  # Calculating class probabilities
        label_counts[(int(row[-1]))] += 1
    if np.amax(label_counts) == len(data):
        cur_node.labelCount = label_counts
        cur_node.dominant = np.argmax(label_counts)
        return
    for label in classes:  # Calculating total entropy
        try:
            entropy -= (label_counts[int(label)] / len(data)) * math.log2((label_counts[int(label)] / len(data)))
        except ValueError:
            entropy -= 0
    best_information_gain = -9999
    best_feature = None
    best_value = None
    for f_index in range(0, len(data[0]) - 1):
        loop_start = np.amin(data[:, f_index])
        loop_end = np.amax(data[:, f_index])
        loop_step = (loop_end - loop_start) / 100

        if loop_start == loop_end:
            continue
        for value in np.arange(loop_start, loop_end, loop_step):  # Loop for optimum information gain
            left, right = split_data(data, f_index, value)
            if len(left) == 0 or len(right) == 0:
                continue
            groups = [left, right]
            aggregate_entropy = split_entropy(groups, classes)
            information_gain = entropy - aggregate_entropy

            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_feature = f_index
                best_value = value
                cur_node.left = Tree()
                cur_node.left.data = np.array(left)
                cur_node.right = Tree()
                cur_node.right.data = np.array(right)
    cur_node.split_col = best_feature  # Store split column and values
    cur_node.split_val = best_value
    if depth + 1 == max_depth:  # Check if max depth reached
        # Do label count, and set dominant label count
        split_label_counts = [0] * (len(np.unique(dataset)))
        for item in cur_node.left.data:
            split_label_counts[int(item[-1])] += 1
        cur_node.left.dominant = np.argmax(split_label_counts)
        cur_node.left.labelCount = split_label_counts
        split_label_counts = [0] * (len(np.unique(dataset)))
        for item in cur_node.right.data:
            split_label_counts[int(item[-1])] += 1
        cur_node.right.labelCount = split_label_counts
        cur_node.right.dominant = np.argmax(split_label_counts)

    if depth + 1 < max_depth:  # If max depth not reached
        if len(cur_node.left.data) != 1:  # Check for nodes with only one sample
            get_best_split(cur_node.left.data, cur_node.left, depth + 1)
        else:
            cur_node.left.dominant = cur_node.left.data[0][-1]
            cur_node.right.dominant = cur_node.right.data[0][-1]
            return
        if len(cur_node.right.data) != 1:
            get_best_split(cur_node.right.data, cur_node.right, depth + 1)
        else:
            cur_node.right.dominant = cur_node.right.data[0][-1]
            cur_node.left.dominant = cur_node.left.data[0][-1]
            return

    else:
        return

    return


'''
    Training section
    Part 2: Bagging and training
'''

'''
    With bagging
'''

# print("->Bagging dataset...")
# k = np.random.randint(8, 12)
# print("\t->", k, "bags")
# arr_size = train_data.shape[0]
# training_size = int(arr_size / k)
# splits = np.arange(0, train_data.shape[0] - training_size, training_size)
# print("->Training model ...")
# roots = []
# for i in range(k):
#     cur_root = tree_creator(train_data[splits[i]:splits[i] + training_size])
#     roots.append(cur_root)
#
# for i in range(k):
#     get_best_split(train_data[splits[i]: splits[i] + training_size], roots[i], 0)

'''
    Without bagging
'''
print("->Training model ...")
set_root = tree_creator(train_data)
get_best_split(train_data, set_root, 0)

'''
    Testing section
'''


def test_acc(test_root):
    row_directions = []
    x_data_final = []
    for data_row in test_data:
        row_direction = []
        _i = 0
        iter_node = test_root
        while 1:

            if iter_node.split_val is None or iter_node.split_col is None:
                data_row = np.append(data_row, iter_node.dominant)
                x_data_final.append(data_row)
                break
            else:
                if data_row[iter_node.split_col] > iter_node.split_val:
                    iter_node = iter_node.right

                    row_direction.append("Right")

                else:

                    iter_node = iter_node.left
                    row_direction.append("Left")
            _i += 1
        row_directions.append(row_direction)

    x_data_final = np.array(x_data_final)

    correct_counter = 0
    for _i in range(len(x_data_final)):
        if x_data_final[_i][-1] == test_data[_i][-1]:
            correct_counter += 1

    correct_counter = correct_counter / len(x_data_final)
    return correct_counter


print("->Testing model ...")

# accuracies = []
# for i in range(k):
#     accuracy = test_acc(roots[i])
#     accuracies.append(accuracy)
# print("[+]Accuracies: ", accuracies)
# print("\t[+]Average accuracy of bags: ", sum(accuracies) / len(accuracies))
# print("\t[+]Min accuracy in bags: ", min(accuracies))
# print("\t[+]Max accuracy in bags: ", max(accuracies))

accuracy = test_acc(set_root)
print("[+]Accuracy: ", accuracy)
