import sys

import numpy as np
import sklearn.metrics


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
    return mac, mic, wei, w, acc


if __name__ == '__main__':
    file = open('labels.txt', 'r')
    my_labels = file.read().split('\n')

    label_addr = sys.argv[1]
    file = open(label_addr, 'r')
    tag = file.read().split('\n')

    cnfs_matrix = make_matrix(my_labels, tag)
    print(cnfs_matrix)
    svm_macro, svm_micro, svm_weighted, report, accu = make_recall_prec_fscore(my_labels, tag)
    print(report)

    print('================================')
    print('accuracy:', accu)
    print('micro', svm_micro)
    print('macro', svm_macro)
    print('weighted', svm_weighted)
