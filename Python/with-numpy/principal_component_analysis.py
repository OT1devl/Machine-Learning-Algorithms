import numpy as np
import matplotlib.pyplot as plt

n_components = 2
mean = None
components = 0

def train(X):
    global mean, components
    mean = np.mean(X, axis=0)
    X = X - mean
    cov = np.cov(X.T)
    eigenvectors, eigenvalues = np.linalg.eig(cov)
    eigenvectors = eigenvectors.T
    idxs = np.argsort(eigenvalues)[::-1]
    eigenvectors, eigenvalues = eigenvectors[idxs], eigenvalues[idxs]
    components = eigenvectors[:n_components]

def transform(X):
    return (X - mean) @ components.T

def generate_data(n_samples=100):
    X = np.random.randn(n_samples, 3)
    return X

def plot_data(X_original, X_transformed):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].scatter(X_original[:, 0], X_original[:, 1], c='blue', label='Original Data')
    ax[0].set_title("Original Data (3D)")
    ax[0].set_xlabel("X1")
    ax[0].set_ylabel("X2")
    ax[1].scatter(X_transformed[:, 0], X_transformed[:, 1], c='red', label='Transformed Data')
    ax[1].set_title("Transformed Data with PCA (2D)")
    ax[1].set_xlabel("Component 1")
    ax[1].set_ylabel("Component 2")
    plt.show()

X = generate_data(200)
train(X)
X_transformed = transform(X)
plot_data(X, X_transformed)