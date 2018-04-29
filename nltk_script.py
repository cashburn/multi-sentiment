import csv
import sys
import pandas as pd
import numpy as np
from scipy import spatial
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

train = pd.read_csv('./Headline_Trainingdata.csv', sep=',', quotechar='"')
test = pd.read_csv('./Headline_Testingdata.csv', sep=',', quotechar='"')

sia = SIA()
pol_scores = [0]*len(train['text'])

for i in range(0,len(train['text'])):
    pol_score = sia.polarity_scores(train['text'][i])['compound']
    pol_scores[i] = int(round(2*pol_score + 2))

count_vect = CountVectorizer(stop_words='english')
X_test_counts = count_vect.fit_transform(test['text'])
X_train_counts = count_vect.transform(train['text'])

clf = MLPClassifier(alpha=1, random_state=65)
clf.fit(X_train_counts, train['sentiment'])
clf2 = SVC(probability = True, gamma=2, C=1)
clf2.fit(X_train_counts, train['sentiment'])

vc = VotingClassifier(estimators=[
    ('mlp', clf), ('svc', clf2)
], voting='hard')
vc.fit(X_train_counts, train['sentiment'])
vc_predicted = vc.predict(X_train_counts)

comb_scores = [0] * len(train['text'])
for i in range(0,len(train['text'])):
    comb_scores[i] = int(round(float(vc_predicted[i]+pol_scores[i])/2.0))

mlp_score = clf.score(X_train_counts, train['sentiment'])
svc_score = clf2.score(X_train_counts, train['sentiment'])
vc_score = vc.score(X_train_counts, train['sentiment'])
pol_score = 1 - spatial.distance.cosine(pol_scores, train['sentiment'])
comb_score = 1 - spatial.distance.cosine(comb_scores, train['sentiment'])

print('MLP: ' + str(mlp_score))
print('SVC: ' + str(svc_score))
print('VC: ' + str(vc_score))
print('POL: ' + str(pol_score))
print('Comb: ' + str(comb_score))