import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, derv=False):
    s = 1 / (1 + np.exp(-x))
    if derv: return s * (1 - s)
    return s

def MSE(y, y_pred, derv=False):
    if derv: return 2*(y_pred-y)
    return np.mean((y_pred-y)**2)

def generate_data(x, noise=0.1):
    y = sigmoid(x)
    y += np.random.normal(0, noise, size=x.shape)
    return np.clip(y, 0, 1)

X = np.expand_dims(np.arange(-10, 10, 0.1), axis=-1)
Y = generate_data(X)

plt.scatter(X, Y)
plt.show()

activation = sigmoid
m = np.random.randn(1, 1) * 0.1
b = np.zeros((1, 1))

def predict(x):
    return activation(x @ m + b)

def learn(y, y_pred, x, lr):
    global m, b
    dL = MSE(y, y_pred, derv=True)
    dL *= activation(x, derv=True)
    N = dL.shape[0]

    dm = x.T @ dL / N
    db = dL.sum(axis=0, keepdims=True) / N

    m -= lr * dm
    b -= lr * db


def train(x, y, epochs=1000, batch_size=32, lr=0.01, print_every=0.1):
    for epoch in range(1, epochs+1):
        for batch in range(0, x.shape[0], batch_size):
            x_batch = x[batch:batch+batch_size]
            y_batch = y[batch:batch+batch_size]
            predictions = predict(x_batch)
            learn(y_batch, predictions, x_batch, lr)
        
        if epoch % max(1, int(epochs*print_every)) == 0:
            print(f'Epoch: [{epoch}/{epochs}]> Loss: {MSE(y, predict(x))}')

train(X, Y)

plt.scatter(X, Y)
plt.plot(X, predict(X), color='orange')
plt.show()
