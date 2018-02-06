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

    acc = sklearn.metrics.accuracy_score(np_tag, np_ans)
    mac = sklearn.metrics.precision_recall_fscore_support(np_tag, np_ans, average='macro')
    mic = sklearn.metrics.precision_recall_fscore_support(np_tag, np_ans, average='micro')
    wei = sklearn.metrics.precision_recall_fscore_support(np_tag, np_ans, average='weighted')

    w = sklearn.metrics.classification_report(np_tag, np_ans,
                                              target_names=['class 1', 'class 2', 'class 3', 'class 5', 'class 8',
                                                            'class 10', 'class 11', 'class 13', 'class 16'])
    return acc, mac, mic, wei, w


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

    cnfs_matrix = make_matrix(ans, tag)
    print(cnfs_matrix)
    acc, svm_macro, svm_micro, svm_weighted, report = make_recall_prec_fscore(ans, tag)
    print(report)

    print('================================')

    print('acc', acc)
    print('micro', svm_micro)
    print('macro', svm_macro)
    print('weighted', svm_weighted)

    print('time: ', time.time() - start_time, "s")

    # last output
    # micro(0.5872916666666667, 0.5872916666666667, 0.5872916666666667, None)
    # macro(0.5879644591207307, 0.586932206092008, 0.5826451872508303, None)
    # weighted(0.5890882388987375, 0.5872916666666667, 0.5834035366368086, None)
    # time: 93.74098205566406 s

    # -----------------------------------------------
    #             precision    recall  f1-score   support

    # class 1        0.48      0.37      0.42      1588
    # class 2        0.41      0.53      0.46      1603
    # class 3        0.50      0.45      0.48      1623
    # class 5        0.63      0.62      0.63      1563
    # class 8        0.42      0.37      0.39      1549
    # class 10       0.70      0.62      0.66      1626
    # class 11       0.86      0.91      0.89      1598
    # class 13       0.70      0.60      0.65      1675
    # class 16       0.58      0.80      0.68      1575
# avg / total       0.59      0.59      0.58     14400
