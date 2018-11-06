import pandas as pd
import numpy as np


def load_train():
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
