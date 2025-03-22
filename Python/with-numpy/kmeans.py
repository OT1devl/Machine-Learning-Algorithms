import numpy as np
import matplotlib.pyplot as plt

K = 2
clusters = [[] for _ in range(K)]
centroids = []
epochs = 100
plot_step = True

def plot(X):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['tab:blue', 'tab:orange']
    
    for i, indices in enumerate(clusters):
        points = X_train[indices]
        ax.scatter(points[:, 0], points[:, 1],
                   s=50, color=colors[i],
                   label=f'Cluster {i}', alpha=0.6, edgecolors='k')
    
    centroids_arr = np.array(centroids)
    ax.scatter(centroids_arr[:, 0], centroids_arr[:, 1],
               s=200, marker='X', color='black',
               label='Centroides')
    
    ax.set_title("K-Means Clustering", fontsize=16)
    ax.set_xlabel("Feature 1", fontsize=14)
    ax.set_ylabel("Feature 2", fontsize=14)
    ax.legend()
    ax.grid(True)
    plt.show()

def generate_data(n_samples=50, offset=2):
    cluster0 = np.random.randn(n_samples, 2) - offset
    cluster1 = np.random.randn(n_samples, 2) + offset
    X = np.vstack((cluster0, cluster1))
    return X

X_train = generate_data()

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

def is_converged(centroids_old, centroids):
    distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(K)]
    return sum(distances) == 0

def closest_centroid(sample, centroids):
    distances = [euclidean_distance(sample, point) for point in centroids]
    closest_idx = np.argmin(distances)
    return closest_idx


def create_clusters(X, centroids):
    clusters = [[] for _ in range(K)]
    for idx, sample in enumerate(X):
        centroid_idx = closest_centroid(sample, centroids)
        clusters[centroid_idx].append(idx)
    return clusters

def get_cluster_labels(clusters, n_samples):
    labels = np.empty(n_samples)
    for cluster_idx, cluster in enumerate(clusters):
        for sample_idx in cluster:
            labels[sample_idx] = cluster_idx

    return labels

def get_centroids(X, clusters, n_features):
    centroids = np.zeros((K, n_features))
    for cluster_idx, cluster in enumerate(clusters):
        cluster_mean = np.mean(X[cluster], axis=0)
        centroids[cluster_idx] = cluster_mean
    return centroids

def predict(X):
    global centroids, clusters
    n_samples, n_features = X.shape
    random_sample_idxs = np.random.choice(n_samples, K, replace=False)
    centroids = [X[idx] for idx in random_sample_idxs]

    for _ in range(epochs):
        clusters = create_clusters(X, centroids)

        if plot_step:
            plot(X)

        centroids_old = centroids
        centroids = get_centroids(X, clusters, n_features)

        if is_converged(centroids_old, centroids):
            break

        if plot_step:
            plot(X)

    return get_cluster_labels(clusters, n_samples)

predict(X_train)
