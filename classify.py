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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

train = pd.read_csv('./Headline_Trainingdata.csv', sep=',', quotechar='"')
test = pd.read_csv('./Headline_Testingdata.csv', sep=',', quotechar='"')

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

X_test_counts = count_vect.fit_transform(test['text'])

X_train_counts = count_vect.transform(train['text'])

X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#clf1 = LogisticRegression(random_state=1)
#clf2 = SVC(gamma=0.001, C=100.)
#clf3 = GaussianNB()

clf = MLPClassifier(alpha=1)
clf.fit(X_train_counts, train['sentiment'])
clf_log = LogisticRegression()
clf_log.fit(X_train_counts, train['sentiment'])

#vc = VotingClassifier(estimators=[
#    ('lr', clf1)
#], voting='hard')
#vc.fit(X_train_counts, train['sentiment'])

score = clf.score(X_train_counts, train['sentiment'])
print(score)

predicted = clf.predict(X_test_counts)
predicted_log = clf_log.predict(X_test_counts)

print('Diff')
for i in range(0,len(predicted)):
    if predicted[i] != predicted_log[i]:
        print(i)

thefile = open('output.csv', 'w')
thefile.write('id,sentiment\n')
for i in range(0,len(predicted)):
    thefile.write('{},{}\n'.format(i, predicted[i]))
    
