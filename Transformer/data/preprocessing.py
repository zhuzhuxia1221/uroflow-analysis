import arff
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

dimensions = 1
data = []
for i in range(1, dimensions + 1):
    filename = 'TRAIN.arff'
    file = open(filename, "r")
    dataset = arff.load(file)
    dataset = np.array(dataset['data'])
    data.append(dataset[ : , 0 : -1])
data = np.array(data)
data = np.transpose(data, (1, 2, 0))
print(data.shape)
np.save('X_train.npy', data)

label = np.array(dataset[ : , -1])
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(label)
print(integer_encoded.shape)
onehot_encoder = OneHotEncoder(sparse = False)
print(integer_encoded)

np.save('y_train.npy', integer_encoded)
