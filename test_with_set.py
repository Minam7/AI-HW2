import pickle
import sys

import gensim
import hazm
import numpy as np
import sklearn


def stem_data(dat):
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

    pickle_in = open("files/svm.pickle", "rb")
    svm = pickle.load(pickle_in)
    answ = svm.decision_function([tpcs])
    return answ.shape[1]


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
    '''
    print(make_matrix([1, 2, 0, 3, 1, 3], [3, 2, 0, 0, 1, 2]))
    print(make_recall_prec_fscore([1, 2, 0, 3, 1, 3], [3, 2, 0, 0, 1, 2]))
    '''
    model_addr = sys.argv[1]
    ldamodel, dictionary, svc = load_models(model_addr)

    data_addr = sys.argv[2]
    tagger = hazm.POSTagger(model='resources/postagger.model')
    stop_words = open('files/stopwords-fa.txt', 'r', encoding='utf-8').read().split('\n')

    file = open(data_addr, 'r', encoding='utf-8')
    content = file.read().split('\n')

    ans = list()
    for i in range(len(content)):
        ans.append(test_data(content[i]))

    label_addr = sys.argv[3]
    file = open(label_addr, 'r', encoding='utf-8')
    tag = file.read().split('\n')

    for i in range(len(tag)):
        tag[i] = int(tag[i])

    cnfs_matrix = make_matrix(ans, tag)
    print(cnfs_matrix)
    print(make_recall_prec_fscore(ans, tag))
