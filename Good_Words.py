"""
Module: Good_Words.py
Use: Finds the most commonly-used words in captions of pictures in the top 25% of engagements
Last Edited: Akrit Sinha, 06-28-2019
"""
# Packages
import pandas
from collections import Counter

# Import the data
pandas.set_option('display.max_columns', None)
train_df = pandas.read_csv("training_set.csv", encoding="ISO-8859-1")

# Retrieves a set of words from top 75% of post captions
sentence = train_df.drop(columns=['Followers at Posting', 'Created', 'Type'])
sentence = sentence[sentence.Engagements > 7.216282e+05]
sentence = sentence['Description'].tolist()
data_set = []
sentence = [x for x in sentence if (str(x) != 'nan')]
for item in sentence:
    for word in item.split(' '):
        if len(word) > 4:
            data_set.append(word)

# Determines the 50 most common words from set
Counter = Counter(data_set)
most_occur = Counter.most_common(50)
print(most_occur)
