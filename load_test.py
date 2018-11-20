import pandas as pd
import numpy as np


def load_test(features):
    dataset = pd.read_csv("test.csv")
    m = len(dataset)
    features_data = []
    bias = np.array([np.ones(m, )])
    features_data.append(bias)
    for i in range(len(features)):
        features_data.append(load_test_feature(features[i]))

    x = np.vstack(features_data)
    dataset = pd.read_csv("gender_submission.csv")
    y = np.array(dataset['Survived'])
    y = y.reshape((y.size, 1))
    return x, y, dataset['PassengerId']


def load_test_feature(feature_name):
    dataset = pd.read_csv("test.csv")
    m = len(dataset)
    feature = np.array(dataset[feature_name])

    # if the feature is Sex
    if (feature_name == "Sex"):
        for i in range(len(feature)):
            feature[i] = 1 if feature[i] == 'male' else 0

    # if  feature is Age
    if (feature_name == "Age"):
        for i in range(m):
            if np.isnan(feature[i]):
                feature[i] = np.nanmean(feature)

    # if feature is Embarked
    if (feature_name == "Embarked"):
        for i in range(len(feature)):
            if (feature[i] == 'C'):
                feature[i] = 1
            elif (feature[i] == 'S'):
                feature[i] = 2
            else:
                feature[i] = 3

    # feature scaling
    if (feature_name != "Sex"):
        feature = (feature - feature.mean()) / feature.std()

    return feature
