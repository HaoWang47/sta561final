import json
import io
import unicodedata
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

DIAGNOSTICS = True

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

# Load data (train_small.json , train.json)
train_file = io.open('train_small.json', 'r')
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
vect = TfidfVectorizer().fit(X)
tf_x = vect.transform(X)
tf_test = vect.transform(X_test)

# Train
nb_clf = MultinomialNB().fit(tf_x, Y)
svm_clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,
                        n_iter=5, random_state=42).fit(tf_x, Y)

# Print diagnostics
if DIAGNOSTICS:
    print "NB empirical accuracy: %f" % np.mean(nb_clf.predict(tf_x) == Y)
    print "SVM empirical accuracy: %f" % np.mean(svm_clf.predict(tf_x) == Y)

# Output predictions
Y_test = svm_clf.predict(tf_test)
out = io.open('submission.csv', 'w')
out.write(u'id,cuisine\n')
for i in range(len(X_test)):
    out.write('%s,%s\n' % (id_test[i], Y_test[i]))

