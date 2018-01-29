import random
from sklearn import svm
import numpy as np


class SupervisedData:
    def __init__(self, dat, lab):
        self.data = dat
        self.label = lab


if __name__ == '__main__':
    DATA_SIZE = 7200
    TRAIN_SIZE = 4320
    VALIDATION_SIZE = 1440
    TEST_SIZE = 1440

    file = open('data/train.content', 'r', encoding='utf-8')
    content = file.read().split('\n')

    file = open('data/train.label', 'r', encoding='utf-8')
    tag = file.read().split('\n')

    datas = list()
    for i in range(DATA_SIZE + 1):
        datas.append(SupervisedData(content[i], tag[i]))

    random.shuffle(datas)

    train_set = datas[:TRAIN_SIZE]
    validation_set = datas[TRAIN_SIZE + 1:TRAIN_SIZE + VALIDATION_SIZE + 1]
    test_set = datas[TRAIN_SIZE + VALIDATION_SIZE + 1:]

    print(len(train_set))
    print(len(validation_set))
    print(len(test_set))

    # SVM chapter

    X = None  # list of features for each data
    y = [row[len(train_set) - 1] for row in train_set]  # list of labels

    # Create the SVC model object
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C, decision_function_shape='ovr').fit(X, y)
