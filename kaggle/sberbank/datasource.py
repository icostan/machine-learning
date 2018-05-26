import pandas as pd
from utils import log


def load_train_data(columns=[]):
    log('Loading train data...')
    data = pd.read_csv('input/train.csv')
    return data[columns]


def load_test_data(columns=[]):
    log('Loading test data...')
    data = pd.read_csv('input/test.csv')
    return data[columns]


def load_data(data, columns=[]):
    log('Loading data...')
    return data[columns]
