import functools as ft
import itertools as it
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tflearn as tflearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing

# from bs4 import BeautifulSoup

#
# helpers
#


def log(l, t):
    print('')
    print('==> ' + l + ': ' + str(t))


def logn(t):
    print('')
    print('##### ' + t)


def display(x, y):
    log('Display charts...')
    sns.jointplot(x, y)
    plt.show()


#
# data
#
logn('Loading data...')
biology_data = pd.read_csv("data/biology.csv")
cooking_data = pd.read_csv("data/cooking.csv")
crypto_data = pd.read_csv("data/crypto.csv")
diy_data = pd.read_csv("data/diy.csv")
robotics_data = pd.read_csv("data/robotics.csv")
travel_data = pd.read_csv("data/travel.csv")
test_data = pd.read_csv("data/test.csv")

data = biology_data
data = data.append(cooking_data)
data = data.append(crypto_data)
data = data.append(diy_data)
data = data.append(robotics_data)
data = data.append(travel_data)

# data.info()
# data.plot()
# plt.show()

#
# feature engineering
#
logn('Feature extraction...')

tf.reset_default_graph()

# add tag columns
data['tag'] = data.tags.map(lambda t: t.split()[0].split('-')[0])


vectorizer = TfidfVectorizer(
    sublinear_tf=True, max_df=0.5, stop_words='english')
vectorizer.fit(data.title)
X = vectorizer.transform(data.title)
log('X shape', X.shape)
# log('STOPWORDS', x_vectorizer.get_stop_words())
# log('X features', str(len(x_vectorizer.get_feature_names())))

le = preprocessing.LabelEncoder()
le.fit(data.tag)
Y = le.transform(data.tag)
log('Y shape', Y.shape)
# log('STOPWORDS', str(y_vectorizer.get_stop_words()))
# log('Y features', str(len(y_vectorizer.get_feature_names())))

# train and test data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42)

clf = MultinomialNB(alpha=.01)
log('Fit', clf.fit(X_train, Y_train))
log('Score', clf.score(X_test, Y_test))

# DNN model
# net = tflearn.input_data(shape=[None, X.shape[1]])
# net = tflearn.fully_connected(net, X.shape[1] * 2)
# net = tflearn.fully_connected(net, Y.shape[1], activation='softmax')
# net = tflearn.regression(net)
# model = tflearn.DNN(net)

# op = ''
# if __name__ == '__main__':
#     if op == 'train':
#         model.fit(X_train, Y_train, validation_set=(X_test, Y_test), n_epoch=10,
#                   batch_size=1000, show_metric=True)
#         logn('Saving model...')
#         model.save('dnn')
#     else:
#         logn('Loading model...')
#         model.load('dnn')

#     # logn('Evaluate...')
#     # print(model.evaluate(X_test, Y_test))

#     logn('Predicting...')
#     XX = x_vectorizer.transform(test_data.title).toarray()
#     log('X_test', str(XX.shape))
#     YY = pd.DataFrame(model.predict(XX))
#     log('YY', str(YY.shape))

#     # save submmission
#     test_data.tags = YY.idxmax(axis=0)
#     pd.DataFrame(data=test_data, columns=['id', 'tags']).to_csv(
#         'submission.csv', index=False)
