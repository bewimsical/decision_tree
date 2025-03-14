import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature  # Feature index
        self.threshold = threshold  # Threshold for splitting
        self.left = left  # Left subtree
        self.right = right  # Right subtree
        self.value = value  # Leaf node value

    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
    
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_labels = np.unique(y)
        
        if len(unique_labels) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return Node(value=Counter(y).most_common(1)[0][0])
        
        best_feature, best_threshold = self._best_split(X, y, num_features)
        
        if best_feature is None:
            return Node(value=Counter(y).most_common(1)[0][0])
        
        left_idx = X[:, best_feature] < best_threshold
        right_idx = ~left_idx
        left_child = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)
    
    #pruning technique?
    def _best_split(self, X, y, num_features):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        left_y, right_y = y[X_column < threshold], y[X_column >= threshold]
        n, n_left, n_right = len(y), len(left_y), len(right_y)
        if n_left == 0 or n_right == 0:
            return 0
        child_entropy = (n_left / n) * self._entropy(left_y) + (n_right / n) * self._entropy(right_y)
        return parent_entropy - child_entropy
    #also pre-pruning
    def _entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / np.sum(counts)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
##post pruning using reduced error pruning  
    def post_prune(self, X_val, y_val):
        self.root = self._prune_tree(self.root, X_val, y_val)

    def _prune_tree(self, node, X_val, y_val):
        if node.is_leaf():
            return node
        node.left = self._prune_tree(node.left, X_val, y_val)
        node.right = self._prune_tree(node.right, X_val, y_val)
        if node.left.is_leaf() and node.right.is_leaf():
            y_pred_subtree = np.array([self._traverse_tree(x, node) for x in X_val])
            current_accuracy = np.mean(y_pred_subtree == y_val)
            majority_label = Counter([node.left.value, node.right.value]).most_common(1)[0][0]
            y_pred_pruned = np.full_like(y_val, fill_value=majority_label)
            pruned_accuracy = np.mean(y_pred_pruned == y_val)
            if pruned_accuracy >= current_accuracy:
                return Node(value=majority_label)
        return node
    
# Example usage:
if __name__ == "__main__":
    #import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import plot_tree
    from sklearn.inspection import DecisionBoundaryDisplay
    
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #max depth = pre-pruning
    tree = DecisionTree(max_depth=3)
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_test)
    
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy:.2f}")

    #sklearn decision tree
    
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, y_train)
    clf_predictions = clf.predict(X_test)

    accuracy = np.mean(clf_predictions == y_test)
    print(f"Accuracy sklearn: {accuracy:.2f}")

