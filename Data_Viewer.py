"""
Module: Data_Viewer
Use: View a 2D line graph comparing one feature to the label
Last Edited: Akrit Sinha, 06-28-2019
"""
# Packages
import pandas
from datetime import datetime
from textblob import TextBlob
from sklearn import preprocessing
import matplotlib.pyplot as plt

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
X['dTime'] = X['Created'].diff(periods=-1)

plt.plot(X['dTime'], y.values.ravel())
plt.show()
