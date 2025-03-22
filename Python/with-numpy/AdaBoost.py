import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples=100):
    X = np.random.uniform(-5, 5, (n_samples, 2))
    y = np.where(X[:, 0] * X[:, 1] >= 0, 1, -1)
    return X, y

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

X, y = generate_data()

adaboost = AdaBoost(n_clf=50)
adaboost.train(X, y)

y_pred = adaboost.predict(X)

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    y_grid_pred = model.predict(X_grid).reshape(xx.shape)

    plt.contourf(xx, yy, y_grid_pred, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Class 1", color="red", edgecolors='k')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], label="Class -1", color="blue", edgecolors='k')
    plt.legend()
    plt.title("Decision Boundary")
    plt.show()

plot_decision_boundary(adaboost, X, y)
