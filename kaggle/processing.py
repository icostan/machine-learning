import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn import preprocessing as pp
from sklearn import feature_extraction as fe

from utils import log


def onehot(y, title='Y'):
    Y = to_categorical(y)
    log(title, Y.shape)
    return Y


def scale(x):
    scaler = pp.MinMaxScaler()
    scaler.fit(x)
    X = scaler.transform(x)
    X = pd.DataFrame(X, index=x.index, columns=x.columns)
    # log('X', X.shape)
    return X, scaler


def label_encoder(labels, title=''):
    le = pp.LabelEncoder()
    le.fit(labels)
    log(title, len(le.classes_))
    return le


def tfidf_encoder(words, title='corpus', max_features=100, stop_words=[]):
    sw = fe.text.ENGLISH_STOP_WORDS.union(stop_words)
    cv = fe.text.TfidfVectorizer(stop_words=sw, max_features=max_features)
    cv.fit(words)
    log(title, len(cv.get_feature_names()))
    return cv


def count_encoder(words, title='corpus', max_features=100, stop_words=[]):
    sw = fe.text.ENGLISH_STOP_WORDS.union(stop_words)
    cv = fe.text.CountVectorizer(
        stop_words=sw, max_features=max_features, binary=True)
    cv.fit(words)
    log(title + ' features', len(cv.get_feature_names()))
    log(title + ' stops', len(cv.stop_words_))
    return cv
