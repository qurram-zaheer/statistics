import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import math


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


def input_format(input_set):
    return input_set.reshape(input_set.shape[1]*input_set.shape[2]*input_set.shape[3], input_set.shape[0])


def sigmoid(z):
    return 1/(1+np.exp(-z))


def propogation(X, Y, W, b, m):
    A = sigmoid(W.T@X + b)

    J = (-1 / m)*(Y@np.log(A).T + (1-Y)@np.log(1-A).T)
    dw = (1/m)*X@(A-Y).T
    db = (1/m)*(np.sum(A-Y))
    return dw, db, np.squeeze(J)


def regression(X, Y, W, b, m, lr, iterations):
    costs = []
    for i in range(iterations):
        dw, db, J = propogation(X, Y, W, b, m)

        if i == 0:
            W_f = W
            b_f = b
            dwf = dw
        W -= lr*dw
        b -= lr*db

        if i % 100 == 0:
            costs.append(J)

        # Print the cost every 100 training iterations
        if i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, J))

    return (W, b), costs, (dw, db), W_f, b_f, dwf


def predict(X, W, b):
    A = sigmoid(W.T@X + b).flatten()

    A = [1 if i >= 0.5 else 0 for i in A]
    return A


def logistic_nn(train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes):
    X = input_format(train_set_x_orig)/255
    X_t = input_format(test_set_x_orig)/255
    Y = train_set_y
    Y_t = test_set_y

    W = np.zeros((X.shape[0], 1))
    b = 0
    m = X.shape[1]

    (W, b), costs, (dw, db), W_f, b_f, dwf = regression(
        X, Y, W, b, m, lr=0.005, iterations=2000)
    Y_train_p = predict(X, W, b)
    Y_test_p = predict(X_t, W, b)

    print("Train error: ", 100 - np.mean(np.abs(Y_train_p - Y))*100)
    print("Test error: ", 100 - np.mean(np.abs(Y_test_p - Y_t))*100)

    return {
        "W": W,
        "b": b,
        "costs": costs,
        "dw": dw,
        "db": db
    }, W_f, b_f, dwf, X


d, W_f, b_f, dwf, X = logistic_nn(train_set_x_orig, train_set_y,
                                  test_set_x_orig, test_set_y, classes)
