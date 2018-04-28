import random
import math
import csv
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

train = pd.read_csv('./Headline_Trainingdata.csv', sep=',', quotechar='"')
test = pd.read_csv('./Headline_Testingdata.csv', sep=',', quotechar='"')

count_vect = CountVectorizer(stop_words='english')
tfidf_transformer = TfidfTransformer()

X_test_counts = count_vect.fit_transform(test['text'])

X_train_counts = count_vect.transform(train['text'])

X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf1 = LogisticRegression(random_state=3)
clf2 = SVC(probability = True, gamma=2, C=1)
clf3 = MLPClassifier(alpha=1)
clf4 = DecisionTreeClassifier(max_depth=5)
clf5 = SGDClassifier(loss = 'hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=3)
#clf3 = GaussianNB()

clf = MLPClassifier(alpha=1)
clf.fit(X_train_counts, train['sentiment'])
clf_log = LogisticRegression()

clf2.fit(X_train_counts, train['sentiment'])

vc = VotingClassifier(estimators=[
    ('mlp', clf3), ('svc', clf2)
], voting='hard')
vc.fit(X_train_counts, train['sentiment'])

score = clf.score(X_train_counts, train['sentiment'])
print(' MLP: ' + str(score))

score2 = clf2.score(X_train_counts, train['sentiment'])
print('SVC: ' + str(score2))

score3 = vc.score(X_train_counts, train['sentiment'])
print('  VC: ' + str(score3))

predicted = clf.predict(X_test_counts)
predicted2 = clf2.predict(X_test_counts)
predicted3 = vc.predict(X_test_counts)

print('Diff')
for i in range(0,len(predicted)):
    if predicted[i] != predicted2[i]:
        print('{}: ({},{},{})'.format(i, predicted[i], predicted2[i], predicted3[i]))

thefile = open('output.csv', 'w')
thefile.write('id,sentiment\n')
for i in range(0,len(predicted3)):
    thefile.write('{},{}\n'.format(i, predicted3[i]))
    
