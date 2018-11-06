
import numpy as np
from logistic_regression import logistic_regression, predict
from load_train import load_train_pclass_sex
from load_test import load_test_pclass_age
from calculations import accuracy

if __name__ == '__main__':
    x, y = load_train_pclass_sex()
    theta = np.zeros((x.shape[0], 1))
    theta = logistic_regression(x.transpose(), y, theta, 0.1, 500)
    # print(theta)
    x_test, y_test, ids = load_test_pclass_age()
    # print(ids)
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

