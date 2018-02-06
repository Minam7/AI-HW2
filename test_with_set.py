import pickle
import sys

import gensim
import numpy as np
import sklearn


def stem_data(dat):
    words = []
    sent = dat.replace('.', '')
    sent = sent.split(' ')
    for iteme in sent:
        if iteme not in stop_words:
            words.append(iteme)

    return words


def load_models(addr):
    ldamodell = gensim.models.LdaModel.load('files/lda.model')
    dictionaryl = gensim.corpora.Dictionary.load('files/lda_dictionary.dict')
    pickle_in = open(addr, "rb")
    svcl = pickle.load(pickle_in)

    return ldamodell, dictionaryl, svcl


def test_data(dat):
    corpus = dictionary.doc2bow(stem_data(dat))
    tpcs = [0 for x in range(100)]
    lda_topics = ldamodel.get_document_topics(corpus)
    for item in lda_topics:
        tpcs[item[0]] = item[1]

    answ = svm.predict([tpcs])
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
    return mac, mic, wei, w, acc


if __name__ == '__main__':

    model_addr = 'files/svm.pickle'
    ldamodel, dictionary, svm = load_models(model_addr)

    data_addr = sys.argv[1]

    label_addr = sys.argv[2]

    stop_words = open('files/stopwords-fa.txt', 'r', encoding='utf-8').read().split('\n')

    file = open(data_addr, 'r', encoding='utf-8')
    content = file.read().split('\n')
    file.close()

    ans = list()
    log = ''

    for i in range(len(content)):
        x = test_data(content[i])
        ans.append(x)
        log += str(x)
        if i != len(content) - 1:
            log += '\n'

    file = open('data/labels.txt', 'w')
    file.write(log)
    file.close()

    file = open(label_addr, 'r', encoding='utf-8')
    tag = file.read().split('\n')

    for i in range(len(tag)):
        tag.append(tag[i])

    cnfs_matrix = make_matrix(ans, tag)
    print(cnfs_matrix)
    svm_macro, svm_micro, svm_weighted, report, accu = make_recall_prec_fscore(ans, tag)
    print(report)

    print('================================')
    print('accuracy:', accu)
    print('micro', svm_micro)
    print('macro', svm_macro)
    print('weighted', svm_weighted)
