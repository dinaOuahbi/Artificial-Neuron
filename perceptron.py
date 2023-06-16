import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.metrics import accuracy_score, log_loss


def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)

def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def log_loss(A, y):
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))

def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)


def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)

def predict(X, W, b):
    A = model(X, W, b)
    # print(A)
    return A >= 0.5

# first one
def artificial_neuron_v1(X, y, learning_rate = 0.1, n_iter = 100):
    # initialisation W, b
    W, b = initialisation(X)

    Loss = []
    acc = []

    for i in tqdm(range(n_iter)):
        A = model(X, W, b)
        Loss.append(log_loss(A, y))
        y_pred = predict(X, W, b)
        acc.append(accuracy_score(y, y_pred))
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)

    y_pred = predict(X, W, b)

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(Loss, label='Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(acc, label='Acc')
    plt.legend()
    plt.show()

    return (W, b)

# VERSION 2
def artificial_neuron(X_train, y_train, X_test, y_test, learning_rate = 0.1, n_iter = 100):
    # initialisation W, b
    W, b = initialisation(X_train)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    # Learning loop
    for i in tqdm(range(n_iter)):
      A = model(X_train, W, b)

      if i %10 == 0:
        #TRAIN
        train_loss.append(log_loss(A, y_train))
        y_pred = predict(X_train, W, b)
        train_acc.append(accuracy_score(y_train, y_pred))

        #TEST
        A_test = model(X_test, W, b)
        test_loss.append(log_loss(A_test, y_test))
        y_pred = predict(X_test, W, b)
        test_acc.append(accuracy_score(y_test, y_pred))

        # MISE A JOUR
        dW, db = gradients(A, X_train, y_train)
        W, b = update(dW, db, W, b, learning_rate)


    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_loss, label='TRAIN')
    plt.plot(test_loss, label='TEST')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_acc, label='TRAIN')
    plt.plot(test_acc, label='TEST')
    plt.legend()
    plt.show()

    return (W, b)