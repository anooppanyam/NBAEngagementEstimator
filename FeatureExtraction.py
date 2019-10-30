#!/usr/bin/env python
# coding: utf-8

# In[162]:


"""
@author anoopppanyam

"""

import pandas as pd
import numpy as np
import spacy
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from spacy.matcher import Matcher
from sklearn.base import TransformerMixin


# In[163]:


#Read and split training set
trainingdf = pd.read_csv('training_set.csv', encoding="ISO-8859-1")
X_train = trainingdf.drop(['Engagements'], axis=1)
Y_train = trainingdf.iloc[:, 0]


# In[32]:



# Create Temporal Features
X_train['Created'] = pd.to_datetime(X_train['Created'], format='%m/%d/%Y %I:%M:%S %p')
X_train['Weekday'] = X_train['Created'].dt.weekday
X_train['Hour'] = X_train['Created'].dt.hour
X_train['Post_Type'] = np.where(X_train['Type'] == 'Photo', 1, 0)


# In[140]:


#Add caption length feature 
X_train['Caption_Len'] = len(X_train['Description'])

# Setting up text analysis -- Bag of words

#Filter out punctuation except @ or # and all stop words
punctuation = re.sub('[#@]', '', string.punctuation)
stop_words = spacy.lang.en.stop_words.STOP_WORDS
matcher = Matcher(nlp.vocab)
matcher.add('HASHTAG', None, [{'ORTH': '#'}, {'IS_ASCII': True}])


def tokenize(text):
    nlp = spacy.lang.en.English()
    doc = nlp(text)
    matches = matcher(doc)
    hashtags = []
    for match_id, start, end in matches:
        hashtags.append(doc[start:end])
    for span in hashtags:
        span.merge()
    doc = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in doc]
    doc = [word for word in doc if (word not in stop_words) and (word not in punctuation)]
    return doc

# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()



#Generate features without normalization
ct_vector = CountVectorizer(tokenizer=tokenize, ngram_range=(1,1))
ct_vector.fit_transform(X_train['Description'].apply(lambda x: np.str(x)))


# In[ ]:





# In[ ]:





# In[ ]:




