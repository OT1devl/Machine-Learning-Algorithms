import numpy as np
import matplotlib.pyplot as plt

def MSE(y, y_pred, derv=False):
    if derv: 
        return 2*(y_pred-y)
    return np.mean((y_pred-y)**2)

def sigmoid(x, derv=False):
    if derv: 
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def BCE(y, y_pred, derv=False):
    if derv: 
        return -y/(y_pred+1e-8)+(1-y)/(1-y_pred+1e-8)
    return np.mean(-y*np.log(y_pred+1e-8)+(1-y)*np.log(1-y_pred+1e-8))

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))




class Linear_Regression:
    def __init__(self):
        self.m = np.random.randn(1, 1) * 0.1
        self.b = np.zeros((1, 1))

    def learn(self, y, outp, x, lr):
        dL = MSE(y, outp, derv=True)
        N = dL.shape[0]
        dm = x.T @ dL / N
        db = dL.sum(axis=0, keepdims=True) / N

        self.m -= lr * dm
        self.b -= lr * db

    def train(self, x, y, epochs=100, batch_size=32, lr=0.01, print_every=0.1):
        for epoch in range(1, epochs+1):
            for batch in range(0, x.shape[0], batch_size):
                x_batch = x[batch:batch+batch_size]
                y_batch = y[batch:batch+batch_size]

                predictions = self.predict(x_batch)
                self.learn(y_batch, predictions, x_batch, lr)
            if epoch % max(1, int(epochs*print_every)) == 0:
                print(f'Epoch: [{epoch}/{epochs}]> Loss: {MSE(y, self.predict(x))}')

    def predict(self, x):
        return x @ self.m + self.b



class Logistic_Regression(Linear_Regression):
    def __init__(self, act=sigmoid):
        super().__init__()
        self.act = act

    def learn(self, y, outp, x, lr):
        dL = BCE(y, outp, derv=True)
        dL *= self.act(outp, derv=True)
        N = dL.shape[0]
        
        dm = x.T @ dL / N
        db = dL.sum(axis=0, keepdims=True) / N

        self.m -= lr * dm
        self.b -= lr * db

    def predict(self, x):
        return self.act(x @ self.m + self.b)



class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions

class AdaBoost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    def train(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1/n_samples))
        self.clfs = []

        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')

            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    missclassified = w[y != predictions]
                    error = sum(missclassified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    if error < min_error:
                        min_error = error
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i

            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1 - min_error) / (min_error + EPS))
            predictions = clf.predict(X)

            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)

            self.clfs.append(clf)
    
    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred



class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class Decision_Tree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features

    def train(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idx = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_thresh = self._best_split(X, y, feat_idx)
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)

        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        left_idx, right_idx = self._split(X_column, threshold)

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])

        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r
        information_gain = parent_entropy - child_entropy

        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])

    def _most_common_label(self, y):
        label_counts = np.bincount(y)
        return np.argmax(label_counts)

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])



class Random_Forest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features

    def train(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = Decision_Tree(max_depth=self.max_depth, 
                                 min_samples_split=self.min_samples_split, 
                                 n_features=self.n_features)
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



class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_idx = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_idx]

        label_counts = {}
        for label in k_nearest_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        most_common_label = max(label_counts, key=label_counts.get)
        return most_common_label



class KMeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.clusters = [[] for _ in range(K)]
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) > 0:
                cluster_mean = np.mean(self.X[cluster], axis=0)
                centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.get_cmap("viridis", self.K)
        
        for i, indices in enumerate(self.clusters):
            points = self.X[indices]
            ax.scatter(points[:, 0], points[:, 1],
                       s=50, color=cmap(i),
                       label=f'Cluster {i}', alpha=0.6, edgecolors='k')
        
        centroids_arr = np.array(self.centroids)
        ax.scatter(centroids_arr[:, 0], centroids_arr[:, 1],
                   s=200, marker='X', color='black',
                   label='Centroides')
        
        ax.set_title("K-Means Clustering", fontsize=16)
        ax.set_xlabel("Feature 1", fontsize=14)
        ax.set_ylabel("Feature 2", fontsize=14)
        ax.legend()
        ax.grid(True)
        plt.show()



class SVM:
    def __init__(self, features):
        self.w = np.random.randn(features)
        self.b = 0.0
    
    def learn(self, condition, lr, lambda_param, x_i, y_i):
        if condition:
            self.w -= lr * (2 * lambda_param * self.w)
        else:
            self.w -= lr * (2 * lambda_param * self.w - y_i * x_i)
            self.b -= lr * y_i
    
    def train(self, x, y, epochs, lambda_param, lr):
        for epoch in range(1, epochs+1):
            for idx, x_i in enumerate(x):
                condition = y[idx] * (x_i @ self.w - self.b) >= 1
                self.learn(condition, lr, lambda_param, x_i, y[idx])
            if epoch % (epochs // 10) == 0:
                print(f'Epochs: [{epoch}/{epochs}]')

    def predict(self, x):
        return np.sign(x @ self.w - self.b)



class Naive_Bayes:
    def train(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator



class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        mean_overall = np.mean(X, axis=0)  
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            S_W += (X_c - mean_c).T.dot(X_c - mean_c)

            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            S_B += n_c * (mean_diff).dot(mean_diff.T)
        
        A = np.linalg.inv(S_W).dot(S_B)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvectors = eigenvectors[idxs]
        self.linear_discriminants = eigenvectors[0:self.n_components]

    def transform(self, X):
        return X @ self.linear_discriminants.T



class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        cov = np.cov(X.T)

        eigenvectors, eigenvalues = np.linalg.eig(cov)

        eigenvectors = eigenvectors.T

        idxs = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[:self.n_components]

    def transform(self, X):
        return (X - self.mean) @ self.components.T
