"""
Module: MLP_R-Classifier.py
Use: Creates an MLP classifier to create extra feature (embedded network)
Last Edited: Akrit Sinha, 06-28-2019
"""
# Packages
import pandas
import pickle
from datetime import datetime

from sklearn.metrics import confusion_matrix, classification_report
from textblob import TextBlob
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Import data to pandas data frame
pandas.set_option('display.max_columns', None)
train_df = pandas.read_csv("training_set.csv", encoding="ISO-8859-1")
q1 = train_df.Engagements.quantile(0.125)
q2 = train_df.Engagements.quantile(0.25)
q3 = train_df.Engagements.quantile(0.375)
q4 = train_df.Engagements.quantile(0.5)
q5 = train_df.Engagements.quantile(0.625)
q6 = train_df.Engagements.quantile(0.75)
q7 = train_df.Engagements.quantile(0.875)


def getrank(eng):
    if eng < q1:
        return 1
    elif eng < q2:
        return 2
    elif eng < q3:
        return 3
    elif eng < q4:
        return 4
    elif eng < q5:
        return 5
    elif eng < q6:
        return 6
    elif eng < q7:
        return 7
    else:
        return 8


train_df['Rank'] = train_df['Engagements'].map(getrank)

# Allocate features (X) and labels (y)
X = train_df.iloc[:, 1:-1]
y = train_df.iloc[:, -1:]


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

# Splits the data into training and testing sets, and resolves NaNs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train = X_train.fillna(X_train.mean())
y_train = y_train.fillna(y_train.mean())
X_test = X_test.fillna(X_test.mean())
y_test = y_test.fillna(y_test.mean())

# Scales the feature set for MLP sensitivity
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Creates the Classifier and fits it to training data
mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100, 100), max_iter=10000, learning_rate='adaptive')
mlp.fit(X_train, y_train.values.ravel())

# Pickles MLP for use with MLP_Creator.py
pickle.dump(mlp, open('MLP_RC', 'wb'))
