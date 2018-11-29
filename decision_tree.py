from load import load, load_feature
from sklearn import tree
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
x_train, y_train = load(features, "data/train.csv")
x_test = load(features, "data/test.csv")

clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print(y_predict)
f = open("data/ans.csv", 'w')
f.write("PassengerId,Survived\n")
ids = load_feature('PassengerId', "data/test.csv")
for i in range(418):
    f.write(str(ids[i]))
    f.write(',')
    f.write(str(y_predict[i]))
    f.write('\n')
f.close()
