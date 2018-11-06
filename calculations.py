

def accuracy(hypothesis, truth):
    facts = (hypothesis == truth)
    cnt = 0
    for i in facts:
        if i[0]:
            cnt += 1
    return cnt / len(truth) * 100
