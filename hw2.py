from __future__ import unicode_literals

import pickle
import random
from sklearn import svm
import numpy as np

import gensim
import hazm


class SupervisedData:
    def __init__(self, dat, lab):
        self.data = dat
        self.label = lab


def stem_data(dat):
    normalizer = hazm.Normalizer()
    dat = normalizer.normalize(dat)
    sent = hazm.sent_tokenize(dat)
    words = []

    for s in sent:
        tagged = list(tagger.tag(hazm.word_tokenize(s)))
        new_tag = list(tagged)

        for token in tagged:
            if token[0] in stop_words:
                new_tag.remove(token)
        lemmatizer = hazm.Lemmatizer()
        for token in new_tag:
            stemmed = lemmatizer.lemmatize(token[0], pos=token[1])
            stemmer = hazm.Stemmer()
            stemmed = stemmer.stem(stemmed)
            if len(stemmed) > 0 and ('#' not in stemmed):
                words.append(stemmed)

    return words


if __name__ == '__main__':
    DATA_SIZE = 7200
    TRAIN_SIZE = 4320
    VALIDATION_SIZE = 1440
    TEST_SIZE = 1440

    # code for first time
    # comment this part if you have saved objects
    tagger = hazm.POSTagger(model='resources/postagger.model')
    stop_words = open('files/stopwords-fa.txt', 'r', encoding='utf-8').read().split('\n')

    file = open('data/train.content', 'r', encoding='utf-8')
    content = file.read().split('\n')

    file = open('data/train.label', 'r', encoding='utf-8')
    tag = file.read().split('\n')

    datas = list()
    docs_words = []

    for i in range(DATA_SIZE + 1):
        paraph = content[i]
        datas.append(SupervisedData(content[i], tag[i]))
        docs_words.append(stem_data(content[i]))

    dictionary = gensim.corpora.Dictionary(docs_words)
    dictionary.save('files/lda_dictionary.dict')

    corpus = [dictionary.doc2bow(text) for text in docs_words]

    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=100, id2word=dictionary, passes=50)
    ldamodel.save('files/lda.model')

    for i in range(DATA_SIZE + 1):
        datas[i].data = ldamodel.get_document_topics(dictionary.doc2bow(docs_words[i]))

    # shuffle data for picking train, validation and test data
    random.shuffle(datas)

    train_set = datas[:TRAIN_SIZE]
    validation_set = datas[TRAIN_SIZE + 1:TRAIN_SIZE + VALIDATION_SIZE + 1]
    test_set = datas[TRAIN_SIZE + VALIDATION_SIZE + 1:]

    # saving objects
    pickle_out = open("train_set.pickle", "wb")
    pickle.dump(train_set, pickle_out)
    pickle_out.close()

    pickle_out = open("validation_set.pickle", "wb")
    pickle.dump(validation_set, pickle_out)
    pickle_out.close()

    pickle_out = open("test_set.pickle", "wb")
    pickle.dump(test_set, pickle_out)
    pickle_out.close()

    # comment part above after saving objects for model learning and division
    # loading objects
    pickle_in = open("train_set.pickle", "rb")
    train_set = pickle.load(pickle_in)

    pickle_in = open("validation_set.pickle", "rb")
    validation_set = pickle.load(pickle_in)

    pickle_in = open("test_set.pickle", "rb")
    test_set = pickle.load(pickle_in)

    # SVM chapter

    X = None  # list of features for each data
    y = [row[len(train_set) - 1] for row in train_set]  # list of labels

    # Create the SVC model object
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C, decision_function_shape='ovr').fit(X, y)
