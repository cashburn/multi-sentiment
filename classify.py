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
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from sklearn.semi_supervised import LabelPropagation

train = pd.read_csv('./Headline_Trainingdata.csv', sep=',', quotechar='"')
test = pd.read_csv('./Headline_Testingdata.csv', sep=',', quotechar='"')

count_vect = CountVectorizer(stop_words='english')

X_test_counts = count_vect.fit_transform(test['text'])
X_train_counts = count_vect.transform(train['text'])

clf = MLPClassifier(alpha=1)
clf.fit(X_train_counts, train['sentiment'])

clf2 = SVC(probability = True, gamma=2, C=1)
clf2.fit(X_train_counts, train['sentiment'])

clf3 = DecisionTreeClassifier(random_state = 0)
clf3.fit(X_train_counts, train['sentiment'])

pa = PassiveAggressiveClassifier(random_state=26)
pa.fit(X_train_counts, train['sentiment'])



vc = VotingClassifier(estimators=[
    ('mlp', clf), ('svc', clf2), ('dt', clf3)
], voting='soft')
vc.fit(X_train_counts, train['sentiment'])

predicted = clf.predict(X_test_counts)
predicted2 = clf2.predict(X_test_counts)
predicted3 = clf3.predict(X_test_counts)
predicted4 = pa.predict(X_test_counts)
predicted_vc = vc.predict(X_test_counts)

sia = SIA()
pol_scores = [0]*len(test['text'])

for i in range(0,len(test['text'])):
    pol_score = sia.polarity_scores(test['text'][i])['compound']
    pol_scores[i] = int(round(2*pol_score + 2))

comb_scores = [0] * len(test['text'])
for i in range(0,len(test['text'])):
    comb_scores[i] = int(round(float(predicted_vc[i]+pol_scores[i])/2.0))

print('Diff')
print('(MLP SVC DT SENT) (VC COMB)')
for i in range(0,len(predicted)):
    if predicted[i] != predicted2[i] or predicted2[i] != predicted3[i]:
        print('{}: ({},{},{},{}) ({},{})'.format(i, predicted[i], predicted2[i], predicted3[i], pol_scores[i], predicted_vc[i], comb_scores[i]))

score = clf.score(X_train_counts, train['sentiment'])
score2 = clf2.score(X_train_counts, train['sentiment'])
score3 = clf3.score(X_train_counts, train['sentiment'])
vc_score = vc.score(X_train_counts, train['sentiment'])
print('MLP: ' + str(score))
print('SVC: ' + str(score2))
print(' DT: ' + str(score3))
print(' VC: ' + str(vc_score))

score4 = pa.score(X_train_counts, train['sentiment'])
print('PA: ' + str(score4))

thefile = open('output.csv', 'w')
thefile.write('id,sentiment\n')
for i in range(0,len(predicted3)):
    thefile.write('{},{}\n'.format(i, predicted3[i]))
    
