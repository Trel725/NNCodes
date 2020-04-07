#!/usr/bin/env python
# coding: utf-8

# ## Feedforward NN classifier
#
# This example demonstrate basic work with tensorflow.keras models,
# how to build simple feed-forward NN classifier and use it to classify Raman spectra

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


# Read the sample dataset. It contains four classes of Raman spectra.
# Class label is available in the last column.

data = pd.read_csv("../spectra.csv.gz").values
features = data[:, :-1]  # take everything but last column
labels = data[:, -1]  # take last column


# Encode labels in one-hot encoding
labels_oh = to_categorical(labels)


# Split the dataset into training and validation
# (here named test for brevity) subsets.
x_train, x_test, y_train, y_test = train_test_split(
    features, labels_oh, test_size=0.3)


# Now define dimensionality of model input and output
inp_dim = x_train.shape[1]
out_dim = y_train.shape[1]


# And define the model. Module tensorflow.keras provides a convinent way for model
# definition by passing output of one layer as input to another.

inp = Input(shape=(inp_dim,))
l = Dense(50, activation="relu")(inp)
l = Dense(25, activation="relu")(l)
out = Dense(out_dim, activation="softmax")(l)


# Now create the classifier model, and compile it using crossentropy loss and adam optimizer.
classifier = Model(inp, out)
classifier.compile(loss="categorical_crossentropy",
                   optimizer="adam", metrics=["accuracy"])
classifier.summary()


# Now when we have training and validation datasets and model, we can start training.
train_history = classifier.fit(
    x=x_train, y=y_train, batch_size=64, epochs=20, validation_data=[x_test, y_test])


# Model fastly converges to 100% classification accuracy. Finally, plot the training history

plt.plot(train_history.history['loss'], "-o", label="Loss")
plt.plot(train_history.history['val_loss'], "-o", label="Validation loss")
plt.plot(train_history.history['accuracy'], "-o", label="Accuracy")
plt.plot(train_history.history['val_accuracy'],
         "-o", label="Validation accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.show()
