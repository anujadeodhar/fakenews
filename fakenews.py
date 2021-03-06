# -*- coding: utf-8 -*-
"""FakeNews.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16YLIAlSs0Pb1kQtnxo4qIKUgwFGHYPwa
"""

from google.colab import drive
# drive.mount('/content/drive/')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/My\ Drive/

import pandas as pd

df = pd.read_csv('C:/Users/Anuja/Downloads/train.csv/train.csv')
#df

X = df.drop('label', axis=1)
#X

df = df.dropna()
#df

messages = df.copy()
messages.reset_index(inplace=True)
#messages

#!pip install PorterStemmer

#!pip install nltk

import nltk
from nltk import *

#nltk.download()



ps = PorterStemmer()
corpus = []

from nltk.corpus import stopwords

# review = re.sub('[^a-zA-Z]', ' ', messages['title'][0])
# print(review)
# review = review.lower()
# print(review)
# review = review.split()
# print(review)
# review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
# print(review)
# review = ' '.join(review)
# print(review)

for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)



#corpus

import sklearn
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, ngram_range=(1, 3))

X = cv.fit_transform(corpus).toarray()
#X

import pickle

# Creating a pickle file for the CountVectorizer
pickle.dump(cv, open('cv-transform.pkl', 'wb'))

from sklearn.model_selection import train_test_split

# Model Building
y = messages['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

#X_train

#y_train

from sklearn.naive_bayes import MultinomialNB

# Fitting Naive Bayes to the Training set
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Creating a pickle file for the Multinomial Naive Bayes model
filename = 'fake-news-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))









# Anuja testing
from sklearn.metrics import accuracy_score

# Load the Model back from file
#with open(filename, 'rb') as file:
#    fake_news_detection_model = pickle.load(file)

#predicting :
#y_pred=fake_news_detection_model .predict(X_test)
#y_pred

#Calculate the accuracy of our model
#accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

#Print the accuracy
#print("Accuracy: {:.2f}%".format(accuracy*100))