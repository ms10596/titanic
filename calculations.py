import numpy as np


def accuracy(hypothesis, truth):
    facts = (hypothesis == truth)
    # print(len(facts), len(hypothesis))
    cnt = 0
    for i in facts:
        # print(i[0])
        if i[0]:
            cnt += 1
    return cnt / len(truth) * 100
