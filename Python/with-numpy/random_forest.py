import numpy as np
import matplotlib.pyplot as plt
from decision_tree import Decision_Tree, generate_data, plot

class Random_Forest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features

    def train(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = Decision_Tree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.train(X_sample, y_sample)
            self.trees.append(tree)

    def _most_common_label(self, y):
        label_counts = np.bincount(y)
        return np.argmax(label_counts)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions
    
X, y = generate_data()
forest = Random_Forest(min_samples_split=2, max_depth=10, n_features=2)
forest.train(X, y)
predictions = forest.predict(X)

plot(X, y, forest)
