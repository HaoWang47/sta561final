import json
import io
import unicodedata
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

DIAGNOSTICS = True

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

# Load data (train_small.json , train.json)
json_file = io.open('train_small.json', 'r')
json_data = json.loads(strip_accents(json_file.read()))

X = []
Y = []
for o in json_data:
    X.append(" ".join(o['ingredients']))
    Y.append(o['cuisine'])

# Print diagnostics
if DIAGNOSTICS:
    print len(X), '=', len(Y)
    print X[0], ':', Y[0]

# Train naive Bayes
nb_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])
nb_clf = nb_clf.fit(X, Y)
print "Naive bayes accuracy: ", np.mean(nb_clf.predict(X) == Y)     

# Train SVM
svm_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                        alpha=1e-3, n_iter=5, random_state=42)),
])
svm_clf = svm_clf.fit(X, Y)
print "SVM accuracy: ", np.mean(svm_clf.predict(X) == Y)  