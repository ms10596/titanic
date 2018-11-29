import numpy as np
import pandas as pd


def load(features, file_name):
    dataset = pd.read_csv(file_name)

    m = len(dataset)
    features_data = []
    bias = np.array([np.ones(m, )])
    features_data.append(bias)
    for i in range(len(features)):
        features_data.append(load_feature(features[i], file_name))

    x = np.vstack(features_data)
    if file_name == "test.csv":
        return x
    y = np.array(dataset['Survived'])
    y = y.reshape((891, 1))
    return x, y


# load specific feature from the training dataset
def load_feature(feature_name, file_name):
    # print(file_name)
    dataset = pd.read_csv(file_name)
    m = len(dataset)
    feature = np.array(dataset[feature_name])

    # if the feature is Sex
    if (feature_name == "Sex"):
        for i in range(len(feature)):
            feature[i] = 1 if feature[i] == 'male' else 0

    # if  feature is Age
    if (feature_name == "Age"):
        avg = np.nanmean(feature)
        for i in range(m):
            if np.isnan(feature[i]):
                feature[i] = avg
            feature[i] = 1 if feature[i] > avg else 0

    # if feature is Embarked
    if (feature_name == "Embarked"):
        for i in range(len(feature)):
            if (feature[i] == 'C'):
                feature[i] = 1
            elif (feature[i] == 'S'):
                feature[i] = 2
            else:
                feature[i] = 3
    if feature_name == "Cabin":
        visited = []
        for i in range(len(feature)):
            if feature[i] in visited:
                feature[i] = visited.index(feature[i])
            else:
                visited.append(feature[i])
                feature[i] = visited.index(feature[i])
    if feature_name == "Fare":
        avg = np.mean(feature)
        for i in range(len(feature)):
            feature[i] = 1 if feature[i] > avg else 0

    # feature scaling
    # if (feature_name != "Sex" and feature_name != "PassengerId"):
    #     feature = (feature - feature.mean()) / feature.std()

    return feature
