import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

data = pd.read_csv('data/train.csv')
print(data.info())
# print(data.corr())

# drop irrelevant/redundant columns
data.drop(['PassengerId', 'Name', 'Fare',
           'Ticket', 'Cabin'], axis=1, inplace=True)
print(data.info())

# to numeric
data['Sex'].replace(['female', 'male'], [0, 1], inplace=True)
data['Embarked'].replace(['C', 'Q', 'S'], [0, 1, 2], inplace=True)

# final data
print(data.head())

X_train = data.values[:, 1:6]
Y_train = data.values[:, 0]

print(X_train.shape)
print(Y_train.shape)

print(plt.figure(figsize=(15, 8)))
