import numpy as np
import pandas as pd
import tflearn
from tflearn.data_utils import to_categorical

# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv

# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
data, labels = load_csv('data/train.csv', target_column=1,
                        columns_to_ignore=[0, 3, 8, 9, 10, 11],
                        categorical_labels=True, n_classes=2)


def preprocess(data):
    for i in range(len(data)):
        # Converting 'sex' field to float (id is 1 after removing labels
        # column)
        data[i][1] = 1.0 if data[i][1] == 'female' else 0.
        for j in range(5):
            data[i][j] = 0 if data[i][j] == '' else data[i][j]
    return np.array(data, dtype=np.float32)


# Preprocess data
data = preprocess(data)

print(pd.DataFrame(data).info())
print(pd.DataFrame(data).describe())

# Build neural network
net = tflearn.input_data(shape=[None, 5])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)

# Let's create some data for DiCaprio and Winslet
dicaprio = [3, 'male', 19, 0, 0]
winslet = [1, 'female', 17, 1, 2]
# Preprocess data
dicaprio, winslet = preprocess([dicaprio, winslet])
# Predict surviving chances (class 1 results)
pred = model.predict([dicaprio, winslet])
print("DiCaprio Surviving Rate:", pred[0][1])
print("Winslet Surviving Rate:", pred[1][1])
