#!/usr/bin/env python
# coding: utf-8

# Convolutional classifier

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from custom_utils import *


# Enable memory growth for GPU,
# it is needed on some systems for convolution to work.

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# Read the sample dataset. It contains four classes of Raman spectra.
# Class labels are available in the last column.

data = pd.read_csv("../spectra.csv.gz").values
features = data[:, :-1]  # take everything but last column
labels = data[:, -1]  # take last column


# Define an auxilary function to generate random polynomials,
# which will simulate fluorescent background


def random_polynomial(x, a=-1, b=1, order=4):
    x_coords = np.random.uniform(x[0], x[-1], size=order)
    x_coords = np.concatenate([x_coords, [x[0], x[-1]]])
    y_coords = np.random.uniform(a, b, size=order + 2)
    coefs = np.polyfit(x_coords, y_coords, deg=order)
    return np.polyval(coefs, x)


# And add the background to the spectra
x = np.arange(features.shape[1])
for i in range(features.shape[0]):
    features[i] += random_polynomial(x)


# Encode labels in one-hot encoding
labels_oh = to_categorical(labels)


# Split the dataset into training and validation (here named test for brevity) subsets.
x_train, x_test, y_train, y_test = train_test_split(
    features, labels_oh, test_size=0.3)


# Now define important constants
inp_dim = x_train.shape[1]
out_dim = y_train.shape[1]

inp = Input(shape=(inp_dim, 1))
# Number of output channels, size of kernel
l = Conv1D(1, 20, activation="relu")(inp)
l = Conv1D(1, 40, activation="relu")(l)
l = Conv1D(1, 80, activation="relu")(l)
# after convolutional layers data must be flattened, i.e.
# converted to 1D vector
l = Flatten()(l)
l = Dense(50, activation="relu")(l)
l = Dense(25, activation="relu")(l)

out = Dense(out_dim, activation="softmax")(l)


conv_classifier = Model(inp, out)
conv_classifier.compile(loss="categorical_crossentropy",
                        optimizer="nadam", metrics=["accuracy"])
conv_classifier.summary()


# We now must add one more dimension to the training data (channel dimension).
# In numpy it is done by simply indexing array with np.newaxis

x_train_chan = x_train[:, :, np.newaxis]
x_test_chan = x_test[:, :, np.newaxis]


# And fit it on the same data
train_history = conv_classifier.fit(
    x=x_train_chan, y=y_train, batch_size=128, epochs=100, validation_data=[x_test_chan, y_test])


# This model converges almost 100% accuracy, which is demonstrates advantages of the convolutional classifiers.
plt.plot(train_history.history['loss'], label="Loss")
plt.plot(train_history.history['val_loss'], label="Validation loss")
plt.plot(train_history.history['accuracy'], label="Accuracy")
plt.plot(train_history.history['val_accuracy'], label="Validation accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.show()
