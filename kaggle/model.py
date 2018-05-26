import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, History
from keras.wrappers.scikit_learn import KerasRegressor

import tensorflow as tf
import tflearn


def keras_single_classification(no_features):
    model = Sequential()
    model.add(Dense(units=int(1.5 * no_features), input_dim=no_features))
    model.add(Activation('relu'))
    model.add(Dense(units=int(no_features / 2)))
    model.add(Activation('relu'))
    model.add(Dense(units=1))
    model.add(Activation('sigmoid'))
    return model


def keras_multi_classification(no_features, no_labels):
    model = Sequential()
    model.add(Dense(units=no_features, input_dim=no_features))
    model.add(Activation('relu'))
    model.add(Dense(units=no_labels))
    model.add(Activation('softmax'))
    return model


def tflearn_classification(no_features, no_labels):
    tf.reset_default_graph()

    net = tflearn.input_data(shape=[None, no_features])
    net = tflearn.fully_connected(net, 2 * no_features)
    net = tflearn.fully_connected(net, no_labels, activation='softmax')
    net = tflearn.regression(net, optimizer='rmsprop',
                             metric='accuracy', loss='categorical_crossentropy')
    model = tflearn.DNN(net)
    return model
