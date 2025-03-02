import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold = None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class Decision_Tree:
    def __init__(self, max_depth = None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, x, y):
        self.root = self._grow_tree(x,y)
    
    def _grow_tree(self, x, y, depth = 0):
        num_samples, num_features = x.shape
        unique_labels = np.unique(y)

        if len(unique_labels) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return Node(value = Counter(y).most_common(1)[0][0])
        
        best_feature, best_threshold = self._best_split(x, y, num_features)

        if best_feature is None:
            return Node(value = Counter(y).most_common(1)[0][0])
        
        left_index = x[:, best_feature] < best_threshold
        right_index = ~left_index
        left_child = self._grow_tree(x[left_index], y[right_index], depth+1)
        right_child = self._grow_tree(x[right_index], y[right_index])

        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)
    
    def _best_split(self, x,y,num_features):
        best_gain = -1
        best_feature = None
        best_threshold = None