import json
import io
import unicodedata
import sys
import numpy as np
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV

DIAGNOSTICS = True
CUTOFF = 35000

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

# Load data (train_small.json , train.json)
train_file = io.open('train.json', 'r')
train_json = json.loads(strip_accents(train_file.read()))

X = []
Y = []
for o in train_json:
    X.append(" ".join(o['ingredients']))
    Y.append(o['cuisine'])

X1 = X[:CUTOFF]
X2 = X[CUTOFF:]
Y1 = Y[:CUTOFF]
Y2 = Y[CUTOFF:]

# Print diagnostics
if DIAGNOSTICS:
    print '%d = %d' % (len(X), len(Y))
    print '%s : %s' % (X[0], Y[0])
    print '%d + %d = %d' % (len(X1), len(X2), len(X))

# TfidfVectorizer
vect = TfidfVectorizer().fit(X)
tf_1 = vect.transform(X1)
tf_2 = vect.transform(X2)

# Train and predict
svm_clf = SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-4,
                        n_iter=5, random_state=42).fit(tf_1, Y1)
Y1_hat = svm_clf.predict(tf_1)
Y2_hat = svm_clf.predict(tf_2)

# Print diagnostics
if DIAGNOSTICS:
    print tf_1.shape
    #print vect.vocabulary_
    print "SVM empirical accuracy: %f" % np.mean(Y1_hat == Y1)
    print "SVM generalization accuracy: %f" % np.mean(Y2_hat == Y2)
    print "%f,%f" % (np.mean(Y1_hat == Y1), np.mean(Y2_hat == Y2))
    print metrics.classification_report(Y2, Y2_hat)
    #print metrics.confusion_matrix(Y2, Y2_hat)
