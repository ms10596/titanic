from load import load, load_feature
import numpy as np
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
x_train, y_train = load(features, "data/train.csv")
# x_test = load(features, "test.csv")
# print(x_train.T[0])
print(load_feature("Ticket", "data/train.csv"))
# clf = tree.DecisionTreeClassifier()
# clf.fit(x_train, y_train)
# y_predict = clf.predict(x_test)
# print(y_predict)
# f = open("ans.csv", 'w')
# f.write("PassengerId,Survived\n")
# for i in range(418):
#     f.write(str(ids[i]))
#     f.write(',')
#     f.write(str(y_predict[i][0]))
#     f.write('\n')
# f.close()
