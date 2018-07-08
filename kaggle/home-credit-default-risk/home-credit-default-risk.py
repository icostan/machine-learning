import sys
import os
sys.path.append(os.path.abspath("./"))

import datasource as ds
import model as m

# feature engineering
matrix, features = ds.load_train_data()
matrix.fillna(0, inplace=True)

# feature selection
Y_train = matrix.pop("TARGET")
print("Y train", Y_train.shape)
X_train = matrix.iloc[:,1:].values
print("X train", X_train.shape)

# keras model
model = m.keras_single_classification(X_train.shape[1])
model.compile(loss='mean_squared_error', optimizer="sgd", metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=10000, epochs=10)
