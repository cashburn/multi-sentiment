import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from scipy import spatial
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from sklearn.linear_model import PassiveAggressiveClassifier

train = pd.read_csv('./Headline_Trainingdata.csv', sep=',', quotechar='"')
test = pd.read_csv('./Headline_Testingdata.csv', sep=',', quotechar='"')

count_vect = CountVectorizer(stop_words='english')
X_test_counts = count_vect.fit_transform(test['text'])
X_train_counts = count_vect.transform(train['text'])

X_train1, X_test1, y_train, y_test = train_test_split(
    train['text'], train['sentiment'], test_size=0.1)

X_test = count_vect.fit_transform(X_test1)
X_train = count_vect.transform(X_train1)

clf = MLPClassifier(alpha=1, random_state=65)
clf.fit(X_train, y_train)

clf2 = SVC(probability = True, gamma=2, C=1)
clf2.fit(X_train, y_train)

clf3 = DecisionTreeClassifier(random_state = 0)
clf3.fit(X_train, y_train)

clf4 = PassiveAggressiveClassifier()
clf4.fit(X_train, y_train)

clf5 = BaggingClassifier(random_state=54)
clf5.fit(X_train, y_train)

clf6 = ExtraTreesClassifier(random_state=0)
clf6.fit(X_train, y_train)

clf7 = GradientBoostingClassifier(random_state=32)
clf7.fit(X_train, y_train)

vc = VotingClassifier(estimators=[
    ('mlp', clf), ('dt', clf3), ('et', clf6), ('bag', clf5), ('grad', clf7)
], voting='soft', weights=[0.3, 0.1, 0.2, 0.1, 0.3])
vc.fit(X_train, y_train)

predicted = clf.predict(X_test)
predicted2 = clf2.predict(X_test)
predicted3 = clf3.predict(X_test)
predicted_vc = vc.predict(X_test)

score1 = clf.score(X_test, y_test)
score2 = clf2.score(X_test, y_test)
score3 = clf3.score(X_test, y_test)
score4 = clf4.score(X_test, y_test)
score5 = clf5.score(X_test, y_test)
score6 = clf6.score(X_test, y_test)
score7 = clf7.score(X_test, y_test)
score_vc = vc.score(X_test, y_test)

sia = SIA()
pol_scores = [0]*len(y_test)

for i in range(0,len(y_test)):
    pol_score = sia.polarity_scores(X_test1.values[i])['compound']
    pol_scores[i] = int(round(2*pol_score + 2))


print('Diff')
print('(MLP SVC DT) (POL, VC)')
count = 0
for i in range(0,len(y_test)):
    if pol_scores[i] != y_test.values[i]:
        count = count + 1
        print(str(pol_scores[i]) + ',' + str(y_test.values[i]))
pol_score = (len(y_test)-count) / len(y_test)

print('MLP: ' + str(score1))
print('SVC: ' + str(score2))
print(' DT: ' + str(score3))
print(' PA: ' + str(score4))
print('Bag: ' + str(score5))
print('Ext: ' + str(score6))
print('Gra: ' + str(score7))
print('POL: ' + str(pol_score))
print(' VC: ' + str(score_vc))

n = 100
maxI = 0
maxN = 0
# for i in range(0, n):
#     clf_new = GradientBoostingClassifier(random_state=i)
#     clf_new.fit(X_train_counts, train['sentiment'])
#     score = clf_new.score(X_train_counts, train['sentiment'])
#     if score > maxN:
#         maxI = i
#         maxN = score
#         print(str(i) + ': ' + str(score))
