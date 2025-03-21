import numpy as np
import matplotlib.pyplot as plt


def generate_data(x, z=0.7):
    return x*np.random.randn() + np.random.randn() + np.random.randn(*x.shape) * z

X = np.expand_dims(np.arange(-10, 10, 0.1), axis=-1)
Y = generate_data(X)


plt.scatter(X, Y)
plt.show()


m = np.random.randn(1, 1) * 0.1
b = np.zeros((1, 1))

def MSE(y, y_pred, derv=False):
    if derv: return 2*(y_pred-y)
    return np.mean((y_pred-y)**2)

def predict(x):
    return x @ m + b

def learn(y, y_pred, x, lr):
    global m, b
    dL = MSE(y, y_pred, derv=True)
    N = dL.shape[0]

    dm = x.T @ dL / N
    db = dL.sum(axis=0, keepdims=True) / N

    m -= lr * dm
    b -= lr * db


def train(x, y, epochs=100, batch_size=32, lr=0.01, print_every=0.1):
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