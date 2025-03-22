import numpy as np
import matplotlib.pyplot as plt

n_samples = 150
n_features = 2
n_classes = 3
n_components = 2

X_class1 = np.random.randn(n_samples, n_features) + np.array([2, 3])
X_class2 = np.random.randn(n_samples, n_features) + np.array([7, 8])
X_class3 = np.random.randn(n_samples, n_features) + np.array([12, 13])

X = np.vstack([X_class1, X_class2, X_class3])
y = np.array([0] * n_samples + [1] * n_samples + [2] * n_samples)

def train(X, y):
    global linear_discriminants
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
    eigenvalues = eigenvalues[idxs]
    eigenvectors = eigenvectors[idxs]
    linear_discriminants = eigenvectors[:n_components]

def transform(X):
    return X @ linear_discriminants.T

def plot(X, X_lda, y):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    axes[0].set_title("Original Data")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")

    axes[1].scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis')
    axes[1].set_title("Data after LDA")
    axes[1].set_xlabel("LD1")
    axes[1].set_ylabel("LD2")

    plt.tight_layout()
    plt.show()

train(X, y)

X_lda = transform(X)

plot(X, X_lda, y)
