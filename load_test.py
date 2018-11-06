import pandas as pd
import numpy as np


def load_test_pclass_age():
    dataset = pd.read_csv("test.csv")
    m = len(dataset)
    x = np.array([np.ones(m, )], dtype=np.float32)
    dataset['Pclass'] = (dataset['Pclass'] - dataset['Pclass'].mean()) / dataset['Pclass'].std()
    p_class = np.array(dataset['Pclass'], dtype=np.float32)
    sex = np.array(dataset['Sex'])
    for i in range(len(sex)):
        sex[i] = 1 if sex[i] == 'male' else 0
    x = np.stack((x[0], p_class, sex))

    dataset = pd.read_csv("gender_submission.csv")
    y = np.array(dataset['Survived'])
    y = y.reshape((y.size, 1))
    return x, y, dataset['PassengerId']

def load_test_sex_pclass_age():
    dataset = pd.read_csv("test.csv")
    m = len(dataset)
    x = np.array([np.ones(m, )], dtype=np.float32)
    dataset['Pclass'] = (dataset['Pclass'] - dataset['Pclass'].mean()) / dataset['Pclass'].std()
    p_class = np.array(dataset['Pclass'], dtype=np.float32)
    sex = np.array(dataset['Sex'])
    for i in range(len(sex)):
        sex[i] = 1 if sex[i] == 'male' else 0
    ages = np.array(dataset['Age'])
    for i in range(m):
        if np.isnan(ages[i]):
            ages[i] = np.nanmean(ages)
    ages = (ages - ages.mean()) / ages.std()
    x = np.stack((x[0], p_class, sex, ages))

    dataset = pd.read_csv("gender_submission.csv")
    y = np.array(dataset['Survived'])
    y = y.reshape((y.size, 1))
    return x, y, dataset['PassengerId']
