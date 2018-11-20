import pandas as pd
import numpy as np
import math


def load_train_pclass_sex():
    dataset = pd.read_csv("train.csv")
    m = len(dataset)
    x = np.array([np.ones(m, )], dtype=np.float32)
    dataset['Pclass'] = (dataset['Pclass'] - dataset['Pclass'].mean()) / dataset['Pclass'].std()
    p_class = np.array(dataset['Pclass'], dtype=np.float32)
    sex = np.array(dataset['Sex'])
    for i in range(len(sex)):
        sex[i] = 1 if sex[i] == 'male' else 0
    x = np.stack((x[0], p_class, sex))
    y = np.array(dataset['Survived'])
    y = y.reshape((891, 1))
    return x, y


def load_train_sex_pclass_age():
    dataset = pd.read_csv("train.csv")
    m = len(dataset)
    x = np.array([np.ones(m, )], dtype=np.float32)

    ages = np.array(dataset['Age'])
    for i in range(m):
        if np.isnan(ages[i]):
            ages[i] = np.nanmean(ages)
    ages = (ages - ages.mean()) / ages.std()
    sex = np.array(dataset['Sex'])
    for i in range(len(sex)):
        sex[i] = 1 if sex[i] == 'male' else 0
    p_class = np.array(dataset['Pclass'], dtype=np.float32)
    x = np.stack((x[0], p_class, sex, ages))
    y = np.array(dataset['Survived'])
    y = y.reshape((891, 1))
    return x, y

#load specific feature from the training dataset
def load_train_feature(feature):
    dataset = pd.read_csv("train.csv")
    m = len(dataset)
    feature = np.array(dataset[feature])

    # if the feature is Sex
    if (feature == "Sex"):
        for i in range(len(feature)):
            feature[i] = 1 if feature[i] == 'male' else 0

    # if  feature is Age
    if (feature == "Age"):
        for i in range(m):
            if np.isnan(feature[i]):
                feature[i] = np.nanmean(feature)

    # if feature is Embarked
    if (feature == "Embarked"):
        for i in range(len(feature)):
            if (feature[i] == 'C'):
                feature[i] = 1
            elif (feature[i] == 'S'):
                feature[i] = 2
            else:
                feature[i] = 3

    # feature scaling
    if (feature != "Sex"):
        feature = (feature - feature.mean()) / feature.std()

    return feature






if __name__ == '__main__':
    load_train()
