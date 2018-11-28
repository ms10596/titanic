import numpy as np

from load import load, load_feature
from logistic_regression import logistic_regression, predict

if __name__ == '__main__':
    features = ["Pclass", "Sex"]
    x, y = load(features, "train.csv")
    theta = np.zeros((x.shape[0], 1))
    theta = logistic_regression(x.transpose(), y, theta, 0.1, 500)
    x_test = load(features, "test.csv")
    y_predict = predict(x_test, theta)
    ids = load_feature("PassengerId", "test.csv")
    # print(ids)
    f = open("ans.csv", 'w')
    f.write("PassengerId,Survived\n")
    for i in range(418):
        f.write(str(ids[i]))
        f.write(',')
        f.write(str(y_predict[i][0]))
        f.write('\n')
    f.close()

    # f = open("features.txt", "a")
    # for i in range(len(features)):
    #     f.write(features[i] + ", ")
    # f.write("\t" + str(accuracy) + "\n")
