import json
import io
import unicodedata
import sys
import numpy as np
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier

#SIZE = 'med'
SIZE = 'lg'
DIAGNOSTICS = True

if SIZE == 'lg':
    CUTOFF = 35000
else:
    CUTOFF = 2500

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

# Load data
if SIZE == 'lg':
    train_file = io.open('train.json', 'r')
else:
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
print "vectorize"
sys.stdout.flush()
vect = TfidfVectorizer(tokenizer=LemmaTokenizer()).fit(X)
tf1 = vect.transform(X1)
tf2 = vect.transform(X2)

# Train and predict
weights = [[20,5], [20, 15], [25, 10], [25,15], [30,10], [30,15]]
for w in weights:
    print "train"
    sys.stdout.flush()
    #clf = AdaBoostClassifier().fit(tf1, Y1)
    #clf = BaggingClassifier().fit(tf1, Y1)
    #clf = KNeighborsClassifier(n_neighbors=8).fit(tf1, Y1)
    #clf = MultinomialNB().fit(tf1, Y1)
    #clf = SVC(kernel='rbf').fit(tf1, Y1)
    clf = RandomForestClassifier(max_depth=w[0], n_estimators=w[1]).fit(tf1, Y1)

    #clf1 = DecisionTreeClassifier(max_depth=40).fit(tf1, Y1)
    #clf3 = SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-4, n_iter=5, random_state=65).fit(tf1, Y1)
    #clf = VotingClassifier(estimators=[('dt', clf1), ('sgd', clf3)], voting='soft', weights=w).fit(tf1, Y1)

    print "predict"
    sys.stdout.flush()
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
