import json
import io
import unicodedata
import sys
import numpy as np
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import RadiusNeighborsClassifier


DIAGNOSTICS = True
CUTOFF = 2500

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

# Load data (train_small.json , train.json)
train_file = io.open('train_medium.json', 'r')
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
vect = TfidfVectorizer(tokenizer=LemmaTokenizer()).fit(X)
tf1 = vect.transform(X1)
tf2 = vect.transform(X2)

# Train and predict
#clf = MultinomialNB().fit(tf1, Y1)
#clf = SVC(kernel='rbf').fit(tf1, Y1)
weights = [[1,1,1]]
for w in weights:
    #clf1 = DecisionTreeClassifier(max_depth=40).fit(tf1, Y1)
    #clf2 = KNeighborsClassifier(n_neighbors=8).fit(tf1, Y1)
    #clf3 = SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-4,
                            #n_iter=5, random_state=65).fit(tf1, Y1)
    #clf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('sgd', clf3)], voting='soft', weights=w).fit(tf1, Y1)
    Y1_hat = clf.predict(tf1)
    Y2_hat = clf.predict(tf2)

    # Print diagnostics
    if DIAGNOSTICS:
        #print tf1.shape
        #print vect.vocabulary_
        print w
        print "Empirical accuracy: %f" % np.mean(Y1_hat == Y1)
        print "Generalization accuracy: %f" % np.mean(Y2_hat == Y2)
        #print "%f,%f" % (np.mean(Y1_hat == Y1), np.mean(Y2_hat == Y2))
        #print metrics.classification_report(Y2, Y2_hat)
        #print metrics.confusion_matrix(Y2, Y2_hat)
