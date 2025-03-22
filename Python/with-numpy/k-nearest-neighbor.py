import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples=50, offset=2):
    X_class0 = np.random.randn(n_samples, 2) - offset
    X_class1 = np.random.randn(n_samples, 2) + offset
    X = np.vstack((X_class0, X_class1))
    y = np.array([0] * n_samples + [1] * n_samples)
    return X, y

X_train, y_train = generate_data()
k = 2

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

def predict(X):
    return np.array([_predict(x) for x in X])

def _predict(x):
    distances = [euclidean_distance(x, x_train) for x_train in X_train]
    k_idx = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_idx]

    label_counts = {}
    for label in k_nearest_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    most_common_label = max(label_counts, key=label_counts.get)
    return most_common_label

x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid = np.c_[xx.ravel(), yy.ravel()]

predictions = predict(grid)
predictions = predictions.reshape(xx.shape)

plt.contourf(xx, yy, predictions, alpha=0.3, cmap='coolwarm')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap='coolwarm')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"KNN (k={k})")
plt.show()
