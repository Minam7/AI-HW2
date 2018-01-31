import logging
import pickle
import time

import numpy as np
import sklearn


class SupervisedData:
    def __init__(self, dat, lab):
        self.data = dat
        self.label = lab
        self.topics = None


def load_models(addr):
    pickle_ins = open(addr, "rb")
    svcl = pickle.load(pickle_ins)

    return svcl


def test_data(dat):
    # arr = np.array(dat.topics).reshape(1, -1)  # these two format are equal but format below is more readable!
    arr = [dat.topics]
    answ = svm.predict(arr)
    # print('answer is : ', answ[0])
    return answ[0]


def make_matrix(predict_data, real_data):
    matr = sklearn.metrics.confusion_matrix(real_data, predict_data)

    return matr


def make_recall_prec_fscore(predict_data, real_data):
    np_ans = np.array(predict_data)
    np_tag = np.array(real_data)
    mac = sklearn.metrics.precision_recall_fscore_support(np_tag, np_ans, average='macro')
    mic = sklearn.metrics.precision_recall_fscore_support(np_tag, np_ans, average='micro')
    wei = sklearn.metrics.precision_recall_fscore_support(np_tag, np_ans, average='weighted')

    return mac, mic, wei


if __name__ == '__main__':
    start_time = time.time()
    DATA_SIZE = 72000
    TRAIN_SIZE = 43200
    VALIDATION_SIZE = 14400
    TEST_SIZE = 14400

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model_addr = 'files/svm.pickle'
    svm = load_models(model_addr)

    pickle_in = open("files/random_datas.pickle", "rb")
    datas = pickle.load(pickle_in)

    train_set = datas[:TRAIN_SIZE]
    validation_set = datas[TRAIN_SIZE + 1:TRAIN_SIZE + VALIDATION_SIZE + 1]
    test_set = datas[TRAIN_SIZE + VALIDATION_SIZE + 1:]

    ans = list()
    log = ''
    for i in range(TEST_SIZE):
        ans.append(test_data(test_set[i]))
        log += str(ans)
        if i != TEST_SIZE - 1:
            log += '\n'

    '''
    file = open('data/labels.txt', 'w', encoding='utf-8')
    file.write(log)

    # label_addr = sys.argv[3]
    '''

    tag = []
    for i in range(TEST_SIZE):
        tag.append(test_set[i].label)

    # cnfs_matrix = make_matrix(ans, tag)
    # print(cnfs_matrix)
    print(make_recall_prec_fscore(ans, tag))

    print('time: ', time.time() - start_time, "s")

    # last output
    # ((0.5879644591207307, 0.586932206092008, 0.5826451872508303, None),
    # (0.5872916666666667, 0.5872916666666667, 0.5872916666666667, None),
    # (0.5890882388987375, 0.5872916666666667, 0.5834035366368086, None))
    # time: 93.52888178825378
