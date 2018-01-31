# ai-hw2
### Artifitial Intelligence HW2

#### text classification in persian

*notice 1: Trained model is not uploaded in git. But saved objects will be saved in files folder in root directory.*

*notice 2: data folder in not uploaded in github, please put train.content and train.label in data folder in root direcotry. In addition resources for hazm is not uploaded either, please put this folder in root directory too.*

* *for training model without stemmer run model.sh*
* *for training model with stemmer run modelstem.sh*
* *for testing model with set run testset.sh*

###### description
* hazm used for stemming, tagging and lemmatizing
* gensim-lda used for extracting features
* sklearn-svm used for supervised learning
* sklearn-confusion matrix used for making confusion matrix and calculating fmeasure, recall, precision