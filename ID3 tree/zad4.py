import numpy as np
import csv
from collections import deque
from sklearn.model_selection import train_test_split

class Node:
    """Node class for ID3 tree."""
    def __init__(self):
        self.value = None
        self.next = None
        self.childs = None

class Id3Tree:
    """ID3 Decision Tree implementation."""
    def __init__(self, X, labels):
        self.X = X
        self.labels = labels
        self.label_classes = list(set(labels))
        self.label_counts = [list(labels).count(x) for x in self.label_classes]
        self.root = None
        self.entropy = self._calculate_entropy([x for x in range(len(self.labels))])

    def _calculate_entropy(self, x_ids):
        """Calculate entropy for given set of data."""
        labels = [self.labels[i] for i in x_ids]
        label_count = [labels.count(x) for x in self.label_classes]
        entropy = sum([-count / len(x_ids) * np.log2(count / len(x_ids)) if count else 0 for count in label_count])
        return entropy

    def _calculate_information_gain(self, x_ids, feature_id):
        """Calculate information gain for a feature."""
        info_gain = self._calculate_entropy(x_ids)
        feature_values = list(set([self.X[x][feature_id] for x in x_ids]))
        feature_counts = [self.X[x][feature_id] for x in x_ids]
        feature_ids = [
            [x_ids[i] for i, x in enumerate(feature_counts) if x == val]
            for val in feature_values
        ]
        info_gain -= sum([len(val_ids) / len(x_ids) * self._calculate_entropy(val_ids)
                          for val_ids in feature_ids])
        return info_gain

    def _get_best_feature(self, x_ids, feature_ids):
        """Find the feature with the highest information gain."""
        feature_entropy = [self._calculate_information_gain(x_ids, feature_id) for feature_id in feature_ids]
        best_feature_id = feature_ids[np.argmax(feature_entropy)]
        return best_feature_id

    def _build_tree(self, x_ids, feature_ids, node):
        """Build ID3 tree recursively."""
        if not node:
            node = Node()
        labels_in_features = [self.labels[x] for x in x_ids]

        # Check if all labels are the same
        if len(set(labels_in_features)) == 1:
            node.value = self.labels[x_ids[0]]
            return node

        # If no more features left
        if len(feature_ids) == 0:
            node.value = max(set(labels_in_features), key=labels_in_features.count)
            return node

        # Choose the best feature
        best_feature_id = self._get_best_feature(x_ids, feature_ids)
        node.value = best_feature_id
        node.childs = []

        feature_values = list(set([self.X[x][best_feature_id] for x in x_ids]))
        for value in feature_values:
            child = Node()
            child.value = value
            node.childs.append(child)
            child_x_ids = [x for x in x_ids if self.X[x][best_feature_id] == value]
            if not child_x_ids:
                child.next = max(set(labels_in_features), key=labels_in_features.count)
            else:
                if feature_ids and best_feature_id in feature_ids:
                    feature_ids.remove(best_feature_id)
                child.next = self._build_tree(child_x_ids, feature_ids, child.next)
        return node

    def build_tree(self):
        """Public method to build the tree."""
        x_ids = list(range(len(self.X)))
        feature_ids = list(range(len(self.X[0])))
        self.root = self._build_tree(x_ids, feature_ids, self.root)

    def _predict_sample(self, sample):
        """Predict the label for a single sample."""
        node = self.root
        while node.childs:
            feature_value = sample[node.value]
            found_child = False
            for child in node.childs:
                if child.value == feature_value:
                    node = child.next
                    found_child = True
                    break
            if not found_child:
                return node.next
        return node.value

    def predict(self, X_test):
        """Predict labels for test data."""
        predictions = [self._predict_sample(sample) for sample in X_test]
        return predictions

def load_data_from_csv(file_path):
    """Load data from CSV file."""
    X = []
    y = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            X.append(row[6:])
            y.append(row[0])  
    return X, y


def main():
    """Main function to run the ID3 decision tree."""
    #file_path = 'breast-cancer.data'
    file_path = 'agaricus-lepiota.data'
    X, y = load_data_from_csv(file_path) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=None)
    classifier = Id3Tree(X_train, y_train)
    classifier.build_tree()
    error_count = sum(1 for i in range(len(X_test)) if classifier.predict([X_test[i]])[0] != y_test[i])
    error_rate = error_count / len(X_test) * 100
    print('Error: {:.2f}%'.format(error_rate))

if __name__ == "__main__":
    main()

