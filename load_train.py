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


if __name__ == '__main__':
    load_train()
