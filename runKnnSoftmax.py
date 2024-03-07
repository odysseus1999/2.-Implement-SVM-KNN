import os
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the CIFAR10 dataset
from keras.datasets import cifar10

# Preprocessing function
def preprocess_data(x_train, x_val, x_test):
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_val -= mean_image
    x_test -= mean_image

    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_val = np.reshape(x_val, (x_val.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))

    x_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
    x_val = np.hstack([x_val, np.ones((x_val.shape[0], 1))])
    x_test = np.hstack([x_test, np.ones((x_test.shape[0], 1))])

    return x_train, x_val, x_test

# Load CIFAR-10 data
baseDir = os.path.dirname(os.path.abspath('__file__')) + '/'
classesName = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_val = x_train[49000:].astype(np.float)
y_val = np.squeeze(y_train[49000:])
x_train = x_train[:49000].astype(np.float)
y_train = np.squeeze(y_train[:49000])
y_test = np.squeeze(y_test)
x_test = x_test.astype(np.float)

# Preprocess the data
x_train, x_val, x_test = preprocess_data(x_train, x_val, x_test)
########################################################################################
                                   # Train kNN classifier
########################################################################################
print("Training kNN classifier...")
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(x_train, y_train)

# Predictions
train_acc = accuracy_score(y_train, knn_classifier.predict(x_train))
val_acc = accuracy_score(y_val, knn_classifier.predict(x_val))
test_acc = accuracy_score(y_test, knn_classifier.predict(x_test))

print('Training Accuracy: {:.2f}%'.format(train_acc * 100))
print('Validation Accuracy: {:.2f}%'.format(val_acc * 100))
print('Test Accuracy: {:.2f}%'.format(test_acc * 100))
