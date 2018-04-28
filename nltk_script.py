import csv
import sys
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

train = pd.read_csv('./Headline_Trainingdata.csv', sep=',', quotechar='"')
test = pd.read_csv('./Headline_Testingdata.csv', sep=',', quotechar='"')

sia = SIA()
results = []

for i in range(0,len(train['text'])):
    pol_score = sia.polarity_scores(train['text'][i])
    print('{},{}'.format(i, str(int(round(2*pol_score['compound'] + 2)))))
