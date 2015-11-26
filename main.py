import json
import io
import unicodedata
import sys
import numpy as np
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

DIAGNOSTICS = True

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

# Load data (train_small.json , train.json)
train_file = io.open('train.json', 'r')
train_json = json.loads(strip_accents(train_file.read()))
test_file = io.open('test.json', 'r')
test_json = json.loads(strip_accents(test_file.read()))

X = []
Y = []
for o in train_json:
    X.append(" ".join(o['ingredients']))
    Y.append(o['cuisine'])

id_test = []
X_test = []
for o in test_json:
    id_test.append(o['id'])
    X_test.append(" ".join(o['ingredients']))

# Print diagnostics
if DIAGNOSTICS:
    print '%d = %d' % (len(X), len(Y))
    print '%s : %s' % (X[0], Y[0])
    print '%d = %d' % (len(X_test), len(id_test))

# TfidfVectorizer
vect = TfidfVectorizer(tokenizer=LemmaTokenizer()).fit(X)
tf_x = vect.transform(X)
tf_test = vect.transform(X_test)

# Train and predict
#nb_clf = MultinomialNB().fit(tf_x, Y)
svm_clf = SGDClassifier(loss='perceptron', penalty='l2', alpha=1e-4,
                        n_iter=5, random_state=42).fit(tf_x, Y)
Y_test = svm_clf.predict(tf_test)

# Print diagnostics
if DIAGNOSTICS:
    #print vect.vocabulary_
    #print "NB empirical accuracy: %f" % np.mean(nb_clf.predict(tf_x) == Y)
    print "SVM empirical accuracy: %f" % np.mean(svm_clf.predict(tf_x) == Y)
    print metrics.classification_report(Y, svm_clf.predict(tf_x))

# Output predictions
out = io.open('submission.csv', 'w')
out.write(u'id,cuisine\n')
for i in range(len(X_test)):
    out.write('%s,%s\n' % (id_test[i], Y_test[i]))

