import numpy as np
from collections import deque


# https://medium.com/datadriveninvestor/easy-implementation-of-decision-tree-with-python-numpy-9ec64f05f8ae
# https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
class DecisionTreeClassifier:
    class Node:
        def __init__(self):
            # the child nodes
            self.right = None
            self.left = None

            # splitting criteria given by the decision tree
            self.column = None
            self.threshold = None

            # probabilities of this nodes decision (determined by splitting criteria)
            self.probabilities = None
            # depth of this node
            self.depth = None

            # it this node is a leaf
            self.is_leaf = False

        def set_child(self, node, is_left_child):
            if is_left_child:
                self.left = node
            else:
                self.right = node

        def set_threshold(self, t):
            self.threshold = t

        def set_probabilities(self, probs):
            self.probabilities = probs

        def set_depth(self, d):
            self.depth = d

        def set_is_leaf(self):
            self.is_leaf = True

        def set_column(self, c):
            self.column = c

    def __init__(self, classes, max_depth=3, min_samples_leaf=1, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.classes = classes

        # Decision tree itself
        self.Tree = None

    def calc_node_probabilities(self, y):
        probabilities = []
        for c in self.classes:
            probability = np.count_nonzero(y == c) / len(y)
            probabilities.append(probability)
        return np.asarray(probabilities)

    def calc_gini_impurity(self, y):
        return 1 - np.sum(self.calc_node_probabilities(y) ** 2)

    def calc_best_split(self, X, y):
        best_split_col = None
        best_thresh = None
        best_info_gain = -999
        impurity_before_split = self.calc_gini_impurity(y)

        # for each column (feature)
        for col_idx in np.arange(X.shape[1]):
            x = X[:, col_idx]
            # test each value in the column as decision threshold
            for threshold in x:
                y_right = y[x >= threshold]
                y_left = y[x < threshold]
                # skip over empty splits
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                # calc the impurity for the new threshold
                impurity_left = self.calc_gini_impurity(y_left)
                impurity_right = self.calc_gini_impurity(y_right)

                information_gain = impurity_before_split
                # apply weighted gini coefficient
                gini = impurity_left * len(y_left) / len(y) + impurity_right * len(y_right) / len(y)
                information_gain -= gini
                if information_gain > best_info_gain:
                    best_split_col = col_idx
                    best_thresh = threshold
                    best_info_gain = information_gain

        if best_info_gain == -999:
            return None, None, None, None, None, None

        # make the split
        x_column = X[:, best_split_col]
        x_left = X[x_column < best_thresh, :]
        x_right = X[x_column >= best_thresh, :]
        y_left = y[x_column < best_thresh]
        y_right = y[x_column >= best_thresh]

        return best_split_col, best_thresh, x_left, y_left, x_right, y_right

    def build_dt(self, X, y):
        stack = [(self.Tree, X, y)]
        while stack:
            node, x_split, y_split = stack.pop(0)

            if node.depth >= self.max_depth:
                node.set_is_leaf()
            if len(X) < self.min_samples_split:
                node.set_is_leaf()
            if len(y) == 1:
                node.set_is_leaf()

            split_col, split_thresh, x_left, y_left, x_right, y_right = self.calc_best_split(x_split, y_split)

            if split_col is None:
                node.set_is_leaf()
            if len(x_left) < self.min_samples_leaf or len(x_right) < self.min_samples_leaf:
                node.set_is_leaf()

            if node.is_leaf:
                continue

            node.set_column(split_col)
            node.set_threshold(split_thresh)

            left = self.Node()
            left.set_depth(node.depth + 1)
            left.set_probabilities(self.calc_node_probabilities(y_left))
            node.set_child(left, True)
            stack.append((left, x_left, y_left))

            right = self.Node()
            right.set_depth(node.depth + 1)
            right.set_probabilities(self.calc_node_probabilities(y_right))
            node.set_child(right, False)
            stack.append((right, x_right, y_right))

            # self.build_dt(x_left, y_left, node.left)
            # self.build_dt(x_right, y_right, node.right)

    def fit(self, X, y):
        self.classes = np.unique(y)

        self.Tree = self.Node()
        self.Tree.set_depth(1)
        self.Tree.set_probabilities(self.calc_node_probabilities(y))

        self.build_dt(X, y)

    def predictSample(self, x, node):
        if node.is_leaf:
            return node.probabilities
        if x[node.column] >= node.threshold:
            probas = self.predictSample(x, node.right)
        elif x[node.column] < node.threshold:
            probas = self.predictSample(x, node.left)
        return probas

    def predict(self, X):
        predictions = []
        for x in X:
            pred = np.argmax(self.predictSample(x, self.Tree))
            predictions.append(pred)
        return np.asarray(predictions)

    def predict_classes(self, X):
        return self.classes[self.predict(X)]


from proj1_helpers import load_csv_data, create_csv_submission

yb_data_train, input_data_train, ids_train = load_csv_data("Data/train.csv", sub_sample=True)
yb_data_test, input_data_test, ids_test = load_csv_data("Data/test.csv", sub_sample=False)

dt = DecisionTreeClassifier(np.unique(yb_data_train), max_depth=5)
dt.fit(input_data_train, yb_data_train)
preds = dt.predict_classes(input_data_test)
create_csv_submission(ids_test, preds, "dec_tree_submission.csv")
