from __future__ import unicode_literals

import logging
import pickle
import random
import time

import gensim
from sklearn import svm

# import sys

dic_data = dict()


class SupervisedData:
    def __init__(self, dat, lab):
        self.data = dat
        self.label = lab
        self.topics = None


def stem_data(dat):
    words = []
    sent = dat.replace('.', '')
    sent = sent.split(' ')
    for iteme in sent:
        if iteme not in stop_words:
            words.append(iteme)

    return words


if __name__ == '__main__':
    start_time = time.time()
    DATA_SIZE = 72000
    TRAIN_SIZE = 43200
    VALIDATION_SIZE = 14400
    TEST_SIZE = 14400

    # comment if its not working
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # data_addr = sys.argv[1]
    # label_addr = sys.argv[2]

    # code for first time
    # comment this part if you have saved objects
    stop_words = open('files/stopwords-fa.txt', 'r', encoding='utf-8').read().split('\n')

    # file = open(data_addr, 'r', encoding='utf-8')
    file = open('data/train.content', 'r', encoding='utf-8')
    content = file.read().split('\n')

    # file = open(label_addr, 'r', encoding='utf-8')
    file = open('data/train.label', 'r', encoding='utf-8')
    tag = file.read().split('\n')

    datas = list()
    docs_words = []

    for i in range(DATA_SIZE + 1):
        paraph = content[i]
        datas.append(SupervisedData(content[i], tag[i]))
        docs_words.append(stem_data(content[i]))

    # saving objects
    pickle_out = open("files/docs_words.pickle", "wb")
    pickle.dump(docs_words, pickle_out)
    pickle_out.close()

    # saving objects
    pickle_out = open("files/datas.pickle", "wb")
    pickle.dump(datas, pickle_out)
    pickle_out.close()

    dictionary = gensim.corpora.Dictionary(docs_words)
    dictionary.save('files/lda_dictionary.dict')

    corpus = [dictionary.doc2bow(text) for text in docs_words]

    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=100, id2word=dictionary, passes=50)
    ldamodel.save('files/lda.model')

    '''
    # load saved objects
    ldamodel = gensim.models.LdaModel.load('files/lda.model')
    dictionary = gensim.corpora.Dictionary.load('files/lda_dictionary.dict')
    
    pickle_in = open("files/datas.pickle", "rb")
    datas = pickle.load(pickle_in)

    pickle_in = open("files/docs_words.pickle", "rb")
    docs_words = pickle.load(pickle_in)
    '''

    for i in range(DATA_SIZE + 1):
        tpcs = [0 for x in range(100)]
        for item in ldamodel.get_document_topics(dictionary.doc2bow(docs_words[i])):
            tpcs[item[0]] = item[1]
            datas[i].topics = tpcs

    # shuffle data for picking train, validation and test data
    random.shuffle(datas)

    # saving objects
    pickle_out = open("files/random_datas.pickle", "wb")
    pickle.dump(datas, pickle_out)
    pickle_out.close()

    '''
    pickle_in = open("files/random_datas.pickle", "rb")
    datas = pickle.load(pickle_in)
    '''

    train_set = datas[:TRAIN_SIZE]
    validation_set = datas[TRAIN_SIZE + 1:TRAIN_SIZE + VALIDATION_SIZE + 1]
    test_set = datas[TRAIN_SIZE + VALIDATION_SIZE + 1:]

    # comment part above after saving objects for model learning and division

    # SVM chapter
    X = [d.topics for d in train_set]  # list of features for each data
    y = [d.label for d in train_set]  # list of labels
    print(X)
    print(y)
    # Create the SVC model object
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='rbf', C=C, decision_function_shape='ovr').fit(X, y)

    # saving svm model
    pickle_out = open("files/svm.pickle", "wb")
    pickle.dump(svc, pickle_out)
    pickle_out.close()

    print('time: ', time.time() - start_time, "s")
