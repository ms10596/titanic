import numpy as np
from numpy import dot


def mse(x, y, theta):
    return -sum(y * np.log(sigmoid(x, theta)) + (1 - y) * np.log(1 - sigmoid(x, theta))) / len(y)


def logistic_regression(x, y, theta, alpha, num_iters):
    for i in range(num_iters):
        a = (sigmoid(x, theta) - y)
        theta = theta - (alpha / y.size) * dot(x.transpose(), a)

    return theta


def predict(x, theta):
    y_predict = np.array(list((map(decide, sigmoid(x.transpose(), theta)))))
    y_predict = y_predict.reshape((418, 1))
    return y_predict


def decide(a):
    if a >= 0.5:
        return 1
    else:
        return 0


def sigmoid(x, theta):
    z = np.array(dot(x, theta), dtype=np.float32)
    temp = np.array(1 / (1 + np.exp(-z)), dtype=np.float32)
    # print(temp.shape)
    return temp
    # print(temp.shape)
