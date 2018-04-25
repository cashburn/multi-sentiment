import random
import math
import csv
import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('./Headline_Trainingdata.csv', sep=',', quotechar='"')
test = pd.read_csv('./Headline_Testingdata.csv', sep=',', quotechar='"')

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

X_test_counts = count_vect.fit_transform(test['text'])

X_train_counts = count_vect.transform(train['text'])

X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#clf = SVC(gamma=0.001, C=100.)
#clf.fit(X_train_counts, train['sentiment'])
clf = LogisticRegression()
clf.fit(X_train_counts, train['sentiment'])
score = clf.score(X_train_counts, train['sentiment'])
print(score)

predicted = clf.predict(X_test_counts)

thefile = open('output.csv', 'w')
thefile.write('id,sentiment\n')
for i in range(0,len(predicted)):
    thefile.write('{},{}\n'.format(i, predicted[i]))
    
