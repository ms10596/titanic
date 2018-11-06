import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from logistic_regression import logistic_regression, predict
from load_train import load_train
from load_test import load_test
from calculations import accuracy

if __name__ == '__main__':
    x, y = load_train()
    theta = np.zeros((3, 1))
    theta = logistic_regression(x.transpose(), y, theta, 0.2, 400)
    # print(theta)
    x_test, y_test, ids = load_test()
    print(ids)
    y_predict = predict(x_test, theta)

    print(accuracy(y_predict, y_test))
    f = open("ans.csv", 'w')
    f.write("PassengerId,Survived\n")
    for i in range(418):
        f.write(str(ids[i]))
        f.write(',')
        f.write(str(y_predict[i][0]))
        f.write('\n')
    f.close()

