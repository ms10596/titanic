import math
from numpy import dot, power, array, concatenate, ones
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def mse(x, y, theta):
    return sum(power((dot(x, theta) - y), 2)) / (2 * len(y))


def linear_regression(x, y, theta, alpha, num_iters):
    for i in range(num_iters):
        theta = theta - (alpha / y.size) * dot((dot(x, theta) - y).transpose(), x)
        # print(theta, mse(x, y, theta))
    return theta
