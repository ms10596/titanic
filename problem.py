import numpy as np
from logistic_regression import logistic_regression, predict
from load_train import load_train
from load_test import load_test
from calculations import accuracy

if __name__ == '__main__':
    features = ["Pclass", "Sex", "Age"]
    x, y = load_train(features)
    theta = np.zeros((x.shape[0], 1))
    theta = logistic_regression(x.transpose(), y, theta, 0.1, 500)
    x_test, y_test, ids = load_test(features)
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
