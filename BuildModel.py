# -*- coding: utf-8 -*-
"""
@author anooppanyam

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


# Read in dataframe and isolate features and target variable 
df = pd.read_csv('training_set.csv', encoding="CP1250")
features = df.drop(['Engagements'], axis=1)
target = df[['Engagements']]

"""
FEATURE EXTRACTION
"""

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
features['Description'] = features['Description'].fillna('')
features[['Length', 'dFollowers', 'dTime']] = features[['Length', 'dFollowers', 'dTime']].fillna(0)


# Add sentiment - create custom extention for spacy tokens
features['Sentiment'] = features['Description'].apply(lambda x : tb(str(x)).sentiment[0])
features['Subjectivity'] = features['Description'].apply(lambda x : tb(str(x)).sentiment[1])


# Setting up text analysis -- Bag of words


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


# Filter out the top 150 scores  --> do not worry about this 

"""
MODELING
"""

# split into train and test models
X_train, X_test, Y_train, Y_test = train_test_split(X, target, test_size=0.25, random_state=55)


# Train and test model....
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn import metrics

# Original model -> accuracy 95.2 verify mape
model = XGBRegressor(learning_rate = 0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective='reg:linear', nthread=4, scale_pos_weight=1, seed=27)

XGBR = XGBRegressor(learning_rate=0.21634371541399422, reg_lambda=2.810294885641067, n_estimators=400, max_depth=8, min_child_weight=19.34404674187281, \
                    objective='reg:linear', nthread=4, reg_alpha=3.2954045033440416, scale_pos_weight=1, seed=27)
 
def xgb_mape(preds, dtrain):
  labels = dtrain.get_label()
  return('mape', -np.mean(np.abs((labels - preds) / (labels+1))))
 
 
def modelfit(alg, X_train, Y_train, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train.values, label=Y_train.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            feval=xgb_mape, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(X_train, Y_train, eval_metric=xgb_mape)
        
    #Predict training set:
    dtrain_predictions = alg.predict(X_train)
    #dtrain_predprob = alg.predict_proba(X_train)[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("R^2 : %.4g" % metrics.r2_score(Y_train.values, dtrain_predictions))
    print("mae Score (Train): %f" % metrics.mean_absolute_error(Y_train.values, dtrain_predictions))
    print("mape error: %.4g" % xgb_mape(dtrain_predictions, xgb.DMatrix(X_train.values, Y_train.values)))
                    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    x,y = zip(*feat_imp[0:20].items())
    plt.ylabel('fscore') 
    plt.xticks(fontsize=7, rotation=35)
    plt.title('Feature Importance')
    plt.bar(x,y)
    plt.show()

modelfit(XGBR, X, target)
modelfit(model, X, target)

import lightgbm as lgbm


LGBM = lgbm.LGBMRegressor(learning_rate = 0.04, num_iterations=5000, max_depth=13, num_leaves=4000, min_child_weight=3, subsample=0.85, colsample_bytree=0.75, \
                      num_thread=4, scale_pos_weight=1, objective='mape', seed=27, max_bin=1000, num_boost_round=1000)

def modelfitlg(alg, X_train, Y_train, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_params()
        xgtrain = lgbm.Dataset(X_train.values, label=Y_train.Engagements)
        cvresult = lgbm.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='mape', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        alg.set_params(n_estimators=cvresult.get('num_iterations'))
    
    #Fit the algorithm on the data
    alg.fit(X_train.values, Y_train.Engagements, verbose=True)
    #model = lgbm.train(alg.get_params(), xgtrain, num_boost_round=1000)
    #alg.fit(X_train, Y_train,eval_metric='mape')
        
    #Predict training set:
    dtrain_predictions = model.predict(X_train)
    #dtrain_predprob = alg.predict_proba(X_train)[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("R^2 : %.4g" % metrics.r2_score(Y_train.values, dtrain_predictions))
    print("mae Score (Train): %f" % metrics.mean_absolute_error(Y_train.values, dtrain_predictions))
    #print("mape error: %.4g" % xgb_mape(dtrain_predictions, lgbm.Dataset(X_train.values, Y_train.values))[1])
                    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    x,y = zip(*feat_imp[0:20].items())
    plt.ylabel('fscore') 
    plt.xticks(fontsize=7, rotation=35)
    plt.title('Feature Importance')
    plt.bar(x,y)
    plt.show()

modelfitlg(LGBM, X, target)


import catboost as cb

CBM = cb.CatBoostRegressor(eval_metric = 'MAPE',learning_rate = 0.4, n_estimators= 415, max_depth=14,  colsample_bylevel=0.75, random_seed = 27)                      
cbm = CBM.fit(X = X, y = target)

pickle.dump(model, open('AWSTunedXGB.dat', "wb"))
pickle.dump(XGBR, open('XGBRModel.dat', "wb"))
pickle.dump(LGBM, open('LGBMModel.dat', 'wb'))
pickle.dump(cbm, open('cbfulldatatrain.sav', 'wb'))
