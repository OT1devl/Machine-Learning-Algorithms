import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples=100, a_line=1, b_line=0, ruido=0.1):
    X = np.random.uniform(-1, 1, (n_samples, 2))
    y = np.sign(X[:, 1] - (a_line * X[:, 0] + b_line) + np.random.uniform(-ruido, ruido, n_samples))
    return X, y

a_line, b_line = 1, 0
X, y = generate_data(100, a_line, b_line)

def step(x):
    return np.where(x > 0, 1, -1)

activation = step
w = np.random.randn(2, 1) * 0.1
bias = np.zeros((1, 1))

def predict(x):
    return activation(x @ w + bias)

def learn(y, y_pred, x, lr):
    global w, bias
    error = y.reshape(-1, 1) - y_pred
    w += lr * x.T @ error
    bias += lr * error.sum(axis=0, keepdims=True)

def train(x, y, epochs=1000, batch_size=32, lr=0.01, print_every=0.1):
    for epoch in range(1, epochs + 1):
        for batch in range(0, x.shape[0], batch_size):
            x_batch = x[batch:batch + batch_size]
            y_batch = y[batch:batch + batch_size]
            predictions = predict(x_batch)
            learn(y_batch, predictions, x_batch, lr)
        if epoch % max(1, int(epochs * print_every)) == 0:
            acc = (predict(x) == y.reshape(-1, 1)).mean()
            print(f'Epoch: [{epoch}/{epochs}]> Accuracy: {acc:.2%}')

train(X, y)

predictions = predict(X)
accuracy = (predictions == y.reshape(-1, 1)).mean()
print(f'Final Accuracy: {accuracy:.2%}')

plt.figure()
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Class -1')
x_vals = np.linspace(-1, 1, 100)
y_vals = -(bias + w[0] * x_vals) / w[1]
plt.plot(x_vals, y_vals.ravel(), 'k--', label='Decision Boundary')
plt.legend()
plt.show()
