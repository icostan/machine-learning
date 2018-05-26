import os
import numpy as np
from sklearn import preprocessing
from keras.preprocessing import image

from utils import log, bytesto
import annotations

INPUT_FOLDER = 'input/'
TRAIN_FOLDER = INPUT_FOLDER + 'train/'
TEST1_FOLDER = INPUT_FOLDER + 'test_stg1/'
TEST2_FOLDER = INPUT_FOLDER + 'test_stg2/'

TYPES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
SIZE = 128

def load_regression_data(categories, size=SIZE, verbose=False):
    x = []
    y = []
    log('Status', 'Processing... ' + str(categories))
    for t in categories:
        folder = TRAIN_FOLDER + t
        files = os.listdir(folder)
        log(t, len(files), suffix='files')
        for filename in files:
            img = load_image(folder + '/' + filename, size=size, expand_dims=False)
            a = annotations.for_image(filename, t)
            if a != None:
                x.append(img)
                y.append([a['x'], a['y']])
    log('Status', 'DONE')

    X = normalize(np.array(x))
    log('X shape', X.shape)
    log('X size', bytesto(X.nbytes, 'm'), suffix='MB')

    Y = np.array(y)
    log('Y shape', Y.shape)
    log('Y size', bytesto(Y.nbytes, 'k'), suffix='KB')

    return X, Y

def load_train_data(categories, size=SIZE, localization=True, verbose=False):
    x = []
    y = []
    log('Status', 'Processing... ' + str(categories))
    for t in categories:
        folder = TRAIN_FOLDER + t
        files = os.listdir(folder)
        log(t, len(files), suffix='files')
        for filename in files:
            img = load_image(folder + '/' + filename, size=size, expand_dims=False)
            x.append(img)
            y.append(t)
    log('Status', 'DONE')

    X = normalize(np.array(x))
    log('X shape', X.shape)
    log('X size', bytesto(X.nbytes, 'm'), suffix='MB')

    Y = preprocessing.LabelEncoder().fit_transform(np.array(y))
    log('Y shape', Y.shape)
    log('Y size', bytesto(Y.nbytes, 'k'), suffix='KB')

    return X, Y


def load_image(path, size=SIZE, expand_dims=True):
    img = image.load_img(path, target_size=(size, size))
    X = image.img_to_array(img)
    if expand_dims:
        X = np.expand_dims(normalize(X), axis=0)
    return X


def load_test1_data():
    return load_test_data(TEST1_FOLDER)


def load_test2_data():
    return load_test_data(TEST2_FOLDER)


def load_test_data(folder):
    t = []
    files = os.listdir(folder)
    log('Status', 'Processing... ' + str(len(files)) + ' files')
    for filename in files:
        path = folder + '/' + filename
        img = load_image(path, expand_dims=False)
        t.append(img)
    log('Status', 'DONE')
    T = normalize(np.array(t))
    log('Shape', T.shape)
    log('Size', bytesto(T.nbytes, 'm'), suffix='MB')
    return T, files


def normalize(X):
    X_train = X.astype('float32')
    X_train /= 255
    return X_train
