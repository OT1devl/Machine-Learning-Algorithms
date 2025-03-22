import numpy as np
import matplotlib.pyplot as plt


def predict(X):
    return np.array([_predict(x) for x in X])

def _predict(x):
    posteriors = []

    for idx, c in enumerate(_classes):
        prior = np.log(_priors[idx])
        posterior = np.sum(np.log(_pdf(idx, x)))
        posterior = posterior + prior
        posteriors.append(posterior)

    return _classes[np.argmax(posteriors)]

def _pdf(class_idx, x):
    mean = _mean[class_idx]
    var = _var[class_idx]
    numerator = np.exp(-((x - mean) ** 2) / (2 * var))
    denominator = np.sqrt(2 * np.pi * var)
    return numerator / denominator

def train(X, y):
    global _classes, n_classes, _mean, _var, _priors
    n_samples, n_features = X.shape
    _classes = np.unique(y)
    n_classes = len(_classes)

    _mean = np.zeros((n_classes, n_features), dtype=np.float32)
    _var = np.zeros((n_classes, n_features), dtype=np.float32)
    _priors = np.zeros(n_classes, dtype=np.float32)

    for idx, c, in enumerate(_classes):
        X_c = X[y == c]
        _mean[idx, :] = X_c.mean(axis=0)
        _var[idx, :] = X_c.var(axis=0)
        _priors[idx] = X_c.shape[0] / float(n_samples)

def generate_data(n_samples=100):
    mean_class_0 = np.array([-2, -2])
    mean_class_1 = np.array([2, 2])
    
    cov = np.array([[1.2, 0.5], [0.5, 1.2]])
    
    X_class_0 = np.random.multivariate_normal(mean_class_0, cov, n_samples // 2)
    X_class_1 = np.random.multivariate_normal(mean_class_1, cov, n_samples // 2)

    X = np.vstack((X_class_0, X_class_1))
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    return X, y

X, y = generate_data()

train(X, y)

plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label="Clase 0", alpha=0.6)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label="Clase 1", alpha=0.6)
plt.legend()
plt.title("Data generated for Naive Bayes")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
