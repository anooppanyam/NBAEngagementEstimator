#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:37:31 2019

@author: anooppanyam
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import spacy
from textblob import TextBlob as tb
from spacy.tokens import Doc
from spacy.matcher import Matcher
import string
import re
from functools import reduce
from operator import and_, or_, contains
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import math
import pickle

# Read in holdout data
hdf = pd.read_csv('holdout_set.csv', encoding="CP1250")

#Isolate features
features = hdf.drop(['Engagements'], axis=1)


""" Preprocess and Feature Extraction """
# Add features and label encode categorical variables 
le = LabelEncoder()
features['Created'] = pd.to_datetime(df['Created'])
features['Weekday'] = features['Created'].dt.weekday
features['Hour'] = features['Created'].dt.hour 
features['Length'] = features.Description.str.len()
features['dFollowers'] = (features['Followers at Posting'].diff(periods= -3)) / (features['Created'].dt.hour*60 + features['Created'].dt.minute).diff(periods=-3)
features['dTime'] = (features['Created'].dt.hour*60 + features['Created'].dt.minute).diff(-3)
features['Type'] = le.fit_transform(features['Type'])
features['Created'] = features['Created'].apply(lambda x : x.timestamp())

# Resolve null values 
features = features.replace([np.inf, -np.inf], np.nan)
features['Description'] = features['Description'].fillna('')
features[['Length', 'dFollowers', 'dTime']] = features[['Length', 'dFollowers', 'dTime']].fillna(0)



# Add sentiment 
features['Sentiment'] = features['Description'].apply(lambda x : tb(str(x)).sentiment[0])
features['Subjectivity'] = features['Description'].apply(lambda x : tb(str(x)).sentiment[1])


#Filter out punctuation except @ or # and all stop wordsh
nlp = spacy.load('en_core_web_lg')
punctuation = re.sub('[#@_]', '', string.punctuation)
stop_words = spacy.lang.en.stop_words.STOP_WORDS
matcher = Matcher(nlp.vocab)
matcher.add('HASHTAG', None, [{'ORTH': '#'}, {'IS_ASCII': True}])
translator = str.maketrans(punctuation, ' '*len(punctuation))
                               
# Filtering out encoding errors 
unwanted = {'–','‘', '’', '“','”','•', '…', '\ufeff1', '\'', '.', '/', '\\'}    
def containsAny(str, set):
    return reduce(or_, map(str.__contains__, set))

# Stemmer 
stemmer = LancasterStemmer() 

"""
Using TFIDF feature extractions
"""

# Aggressively stem and lemmatize to reduce sparcity

def PreprocessAndTokenize(text): 
    # Keep hashtags
    text = re.sub('[?][?]', ' EMOJI ', text)
    text = text.translate(translator)
    doc = nlp(text)
    matches = matcher(doc)
    hashtags = []
    for match_id, start, end in matches:
        hashtags.append(doc[start:end])
    for span in hashtags:
        span.merge()
        # Lemmatize first and then stem 
    doc = [word.lower_.strip() if word.lemma_ == "-PRON-" or word.text.find('@') != -1 else stemmer.stem(word.lemma_.lower().strip()) for word in doc]
    doc = [word for word in doc if (word not in stop_words) and  (word not in punctuation)]
    doc = [word for word in doc if (not containsAny(word, unwanted) and ((word.find('@') != -1) or ((not any(str.isdigit(c) for c in word)) and len(word) >= 4)))]
    return doc
    

# Create a Term-Frequency - Inverse Document Frequency weighted model
tfv = TfidfVectorizer(tokenizer=PreprocessAndTokenize, analyzer='word', ngram_range=[1,1])
X_tf = tfv.fit_transform(features['Description'].apply(lambda x : np.str(x)))
newfeats = pd.DataFrame(X_tf.toarray(), columns=tfv.get_feature_names()) 

# Drop cols 
filteredfeats = features.drop(['Description'], axis=1)

# Normalize Data
normalizer_sparse = preprocessing.StandardScaler(with_mean=False)
normalizer_reg = preprocessing.StandardScaler(with_mean=True)

newftnorm = normalizer_sparse.fit_transform(newfeats)
oldftnorm = normalizer_reg.fit_transform(filteredfeats)

# Create normalized merged dataframe 
X = pd.concat([pd.DataFrame(newftnorm, columns=[newfeats.columns]), pd.DataFrame(oldftnorm, columns=[filteredfeats.columns])], axis=1)

""" ===> RUN THIS PART """

import xgboost as xgb
import lightgbm as lgbm
import catboost as cb
""" MODELING """
# Read in model
model = pickle.load(open('XGBRFull.dat', "rb"))
XGBR = pickle.load(open('xgb200', 'rb'))
LGBM = pickle.load(open('LGBMModel.dat', 'rb'))
CBM = pickle.load(open('cbfulldatatrain.sav', 'rb'))

unwanted = "@njohn13, rebel, prop, tonlead, viol, bogut, @whynotfoundation, @thenbpa, grad, @preedgolf, montcrieff, priceless, pierr, grammy, proof, looooos, excel, @biggame_08, pizz, patterson, mandel, exclam, @naterobinson, insight, @imanshumpert, amad, wedgy, effect, @rhff, @steph, @cassyathenaphoto, @1ingram4, surv, hangtim, sean, consol, @barenakedladiesmusic, display, shan, beep, ayyy, @kellyschu, talkin, trev, vist, sist, cloud, hyperdunk, illustry, @jabari_bird, @fergie, @filayyyy, @iisaiahthomas, convert, gordon, rialto, stockton, regen, cast, perkin, @iammontaellis, control, yodel, americ, recovery, chul, alexand, demo, prid, renit, voorh, jarrel, domest, setup, @complexcon, summerleagu, andrew, math, @divincenzo9, threat, @govballnyc, superhum, @rademita, anticip, @taylorswift, mihja, emon, @jumpman23, compos, @mr_carter5, bingo, rand, courtesy, nelson, soon, troit, carolin, @eddavisxvii, @juliancenter, @t__cloud9, @markeaton7ft4, erick, anthem, grown, @katyperry, addison, ariel, @repyourwork, aaaaaand" 
unwantedlist = unwanted.split(", ")

A = X.drop(unwantedlist, axis=1)
A = A[model.get_booster().get_fscore().keys()]









predictions = 0.60*model.predict(A) + 0.15*XGBR.predict(xgb.DMatrix(A.as_matrix())) + 0.10*LGBM.predict(A.values) + 0.15*CBM.predict(A)

