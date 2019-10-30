"""
Module: MLP_Creator.py
Use: Creates a multilayer perceptron along with testing sets and pickles them for use with MLP_Tester.py
Last Edited: Akrit Sinha, 06-28-2019
"""
# Packages
import pandas
import pickle
from datetime import datetime
from textblob import TextBlob
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Import data to pandas data frame
pandas.set_option('display.max_columns', None)
train_df = pandas.read_csv("training_set.csv", encoding="ISO-8859-1")

# Allocate features (X) and labels (y)
X = train_df.iloc[:, 1:]
y = train_df.iloc[:, :1]


# Returns time of post in terms of hours and minutes
def gettime(st):
    st = st[:-4]
    d = datetime.strptime(st, "%Y-%m-%d %H:%M:%S")
    return int(d.hour * 60 + d.minute)


# Returns post date in terms of month and day of week
def getday(st):
    st = st[:-4]
    d = datetime.strptime(st, "%Y-%m-%d %H:%M:%S")
    return d.isoweekday() + d.month


# Returns sentiment level of post caption
def getsent(st):
    if isinstance(st, str):
        t = TextBlob(st)
        return t.sentiment.polarity
    else:
        return 0


# Cleans the data, applies above functions to create complete numerical data frame
le = preprocessing.LabelEncoder()
X['Type'] = le.fit_transform(X['Type'])
X['Date'] = X['Created'].map(getday)
X['Created'] = X['Created'].map(gettime)
X['dFollowers'] = (X['Followers at Posting'].diff(periods=-3)) / (X['Created'].diff(periods=-3))
X['Sentiment'] = X['Description'].map(getsent)
X['Punctuation'] = X['Description'].str.count('!!!|ebron|rving|urry|iannis|arden|Why') \
                   + 2*X['Description'].str.count('@|#|ames')
X['Description'] = X['Description'].str.len()
X['dTime'] = X['Created'].diff(periods=-3)

# Creates added feature columns from E-Classifier
X_temp = X.fillna(X.mean())
scaler_temp = StandardScaler()
scaler_temp.fit(X_temp)
X_temp = scaler_temp.transform(X_temp)
classifier = pickle.load(open('MLP_EC', 'rb'))
X['Estimate'] = classifier.predict(X_temp)

# Creates added feature columns from R-Classifier
X_temp = X.fillna(X.mean())
scaler_temp = StandardScaler()
scaler_temp.fit(X_temp)
X_temp = scaler_temp.transform(X_temp)
classifier = pickle.load(open('MLP_RC', 'rb'))
X['Rank'] = classifier.predict(X_temp)

# Splits the data into training and testing sets, and resolves NaNs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
X_train = X_train.fillna(X_train.mean())
y_train = y_train.fillna(y_train.mean())
X_test = X_test.fillna(X_test.mean())
y_test = y_test.fillna(y_test.mean())

# Scales the feature set for MLP sensitivity
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Creates the MLP and fits it to training data
mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=75000, learning_rate='adaptive')
mlp.fit(X_train, y_train.values.ravel())

# Pickles MLP and testing sets for use with MLP_Tester
pickle.dump(X_test, open('X_test', 'wb'))
pickle.dump(y_test, open('y_test', 'wb'))
pickle.dump(mlp, open('MLP', 'wb'))
