import pickle

import sys
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import KFold


class SupervisedData:
    def __init__(self, dat, lab):
        self.data = dat
        self.label = lab
        self.topics = None

model_addr = sys.argv[1]
# model_addr = 'files/svm.pickle'
pickle_in = open(model_addr, "rb")
datas = pickle.load(pickle_in)

data_addr = sys.argv[2]
# data_addr = 'files/random_datas.pickle'
pickle_in2 = open(data_addr, "rb")
svc = pickle.load(pickle_in2)

X_digits = [d.topics for d in datas]  # list of features for each data
y_digits = [d.label for d in datas]  # list of labels


k_fold = KFold(n_splits=5)
train_set = []
test_set = []
train_set_y = []
test_set_y = []
''''
for train_indices, test_indices in k_fold.split(X_digits):
     print('Train: %s | test: %s' % (train_indices, test_indices))
     train_set.append([X_digits[train_indice] for train_indice in train_indices])
     test_set.append([X_digits[test_indice] for test_indice in test_indices])
     train_set_y.append([y_digits[train_indice] for train_indice in train_indices])
     test_set_y.append([y_digits[test_indice] for test_indice in test_indices])

'''
#for i in range(5):
#   print(svc.fit(train_set[i], train_set_y[i]).score(test_set[i], test_set_y[i]))

scores = cross_val_score(svc, X_digits, y_digits, cv=5, n_jobs=-1)
print(scores)
average = 0
for s in scores:
    average += s

average /= 5
print(average)