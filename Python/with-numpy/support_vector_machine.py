import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples=100, n_features=2, noise=0.1): 
    X = np.random.randn(n_samples, n_features)
    y = np.sign(X[:, 0] + np.random.normal(0, noise, size=n_samples))
    y = (y + 1) // 2  
    return X, y.reshape(-1, 1)

X, y = generate_data(n_samples=200)
y = np.where(y <= 0, -1, 1)

w = np.random.randn(2)
b = 0.0

def predict(x):
    return np.sign(np.dot(x, w) - b)

def learn(condition, lr, lambda_param, x_i, y_i):
    global w, b
    if condition:
        w -= lr * (2 * lambda_param * w)
    else:
        w -= lr * (2 * lambda_param * w - y_i * x_i)
        b -= lr * y_i

def train(x, y, epochs=1000, lambda_param=0.01, lr=0.001):
    for epoch in range(1, epochs+1):
        for idx, x_i in enumerate(x):
            condition = y[idx] * (np.dot(x_i, w) - b) >= 1
            learn(condition, lr, lambda_param, x_i, y[idx])
        if epoch % (epochs // 10) == 0:
            print(f'Epochs: [{epoch}/{epochs}]')

train(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1
xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], w) - b)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("SVM - Gradient descent")
plt.show()
