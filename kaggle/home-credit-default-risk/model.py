import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, History
from keras.wrappers.scikit_learn import KerasRegressor

def keras_single_classification(no_features, multiplication=2):
    model = Sequential()
    model.add(Dense(units=int(no_features), input_dim=no_features))
    model.add(Activation('relu'))
    model.add(Dense(units=int(multiplication * no_features)))
    model.add(Activation('relu'))
    model.add(Dense(units=int(no_features / multiplication)))
    model.add(Activation('relu'))
    model.add(Dense(units=1))
    model.add(Activation('sigmoid'))
    return model
