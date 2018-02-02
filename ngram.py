from __future__ import unicode_literals

import logging
import pickle
import random
import time
import sys

import gensim
from sklearn import svm

dic_data = dict()


class Ngrams:

    def __init__(self, n_list):
        self.n_list = n_list
        self.indices = {}

    def fit(self, sentence):
        """Magic n-gram function fits to vector indices."""
        i, inp = len(self.indices)-1, sentence.split()
        for n in self.n_list:
            for x in zip(*[inp[i:] for i in range(n)]):
                if self.indices.get(x) == None:
                    i += 1
                    self.indices.update({x: i})

    def transform(self, sentence):
        """Given a sentence, convert to a gram vector."""
        v, inp = [0] * len(self.indices), sentence.split()
        for n in self.n_list:
            for x in zip(*[inp[i:] for i in range(n)]):
                if self.indices.get(x) != None:
                    v[self.indices[x]] += 1
        return v


class SupervisedData:
    def __init__(self, dat, lab):
        self.data = dat
        self.label = lab
        self.ngram = None
        self.text = None


def stem_data(dat):
    words = []
    sent = dat.replace('.', '')
    sent = sent.split(' ')
    for iteme in sent:
        if iteme not in stop_words:
            words.append(iteme)
    text = ' '.join(words)
    ng.fit(text)
    return text


if __name__ == '__main__':
    start_time = time.time()
    DATA_SIZE = 72000
    TRAIN_SIZE = 43200
    VALIDATION_SIZE = 14400
    TEST_SIZE = 14400

    ng = Ngrams(n_list=[1])

    # comment if its not working
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    data_addr = 'data/train.content'
    label_addr = 'data/train.label'

    # code for first time
    # comment this part if you have saved objects
    stop_words = open('files/stopwords-fa.txt', 'r', encoding='utf-8').read().split('\n')

    # file = open(data_addr, 'r', encoding='utf-8')
    file = open('data/train.content', 'r', encoding='utf-8')
    content = file.read().split('\n')
    print("content read \n")

    # file = open(label_addr, 'r', encoding='utf-8')
    file = open('data/train.label', 'r', encoding='utf-8')
    tag = file.read().split('\n')
    print("label read \n")

    datas = list()

    for i in range(DATA_SIZE + 1):
        paraph = content[i]
        datas.append(SupervisedData(content[i], tag[i]))
        datas[i].text = stem_data(content[i])
    print("stem done \n")


    for i in range(DATA_SIZE + 1):
        datas[i].ngram = ng.transform(datas[i].text)
    print("vectorized \n")


    # shuffle data for picking train, validation and test data
    random.shuffle(datas)

    # saving objects
    # saving objects
    pickle_out = open("files/random_datasNgram.pickle", "wb")
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
    X = [d.ngram for d in train_set]  # list of features for each data
    y = [d.label for d in train_set]  # list of labels
    print(X)
    print(y)
    # Create the SVC model object
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='rbf', C=C, decision_function_shape='ovr').fit(X, y)

    # saving svm model
    pickle_out = open("files/svmNgram.pickle", "wb")
    pickle.dump(svc, pickle_out)
    pickle_out.close()

    print('time: ', time.time() - start_time, "s")
