import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('./Headline_Trainingdata.csv', sep=',', quotechar='"')
test = pd.read_csv('./Headline_Testingdata.csv', sep=',', quotechar='"')

count_vect = CountVectorizer(stop_words='english')
train.drop_duplicates(subset='text', keep='first')
X_test_counts = count_vect.fit_transform(test['text'])
X_train_counts = count_vect.transform(train['text'])

n = 100
maxI = 0
maxN = 0
for i in range(0, n):
    clf = DecisionTreeClassifier(random_state=i)
    clf.fit(X_train_counts, train['sentiment'])
    score = clf.score(X_train_counts, train['sentiment'])
    if score >= maxN:
        maxI = i
        maxN = score
        print(str(i) + ': ' + str(score))
