#!/usr/bin/env python
# coding: utf-8

# Load libraries
import numpy as np
from sklearn import metrics
import pandas as pd
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Layer, Input
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Load train features data
X_train = np.load('data/X_train.npy')
X_train = X_train[:10000,:]
print(X_train.shape)

# Load train labels data
y_train = np.load('data/y_train.npy')
y_train = y_train[:10000,:]
print(y_train.shape)

# Load test features data
X_test = np.load('data/X_test.npy')
X_test = X_test[:1000,:]
print(X_test.shape)

# Load test labels
y_test = np.load('data/y_test.npy')
y_test = y_test[:1000,:]
print(y_test.shape)

# RNN architecture
input_data = Input(shape=(1000,4))

## Convolutional Layer
output = Conv1D(320,kernel_size=26,activation="relu")(input_data)
output = MaxPooling1D()(output)
output = Dropout(0.2)(output)

## BiLSTM Layer
output = Bidirectional(LSTM(320,return_sequences=True))(output)
output = Dropout(0.5)(output)

flat_output = Flatten()(output)

## FC Layer
FC_output = Dense(695)(flat_output)
FC_output = Activation('relu')(FC_output)

## Output Layer
output = Dense(690)(FC_output)
output = Activation('sigmoid')(output)

model = Model(inputs=input_data, outputs=output)

print('compiling model')
model.compile(loss='binary_crossentropy', optimizer='adam')

print('model summary')
model.summary()

checkpointer = ModelCheckpoint(filepath="./model/bilstm_model.hdf5", verbose=1, save_best_only=False)
earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

history = model.fit(X_train, y_train, batch_size=100, epochs=2, shuffle=True, verbose=1, validation_split=0.1, callbacks=[checkpointer,earlystopper])

# Metrics
training_loss = np.mean(history.history['loss'])
print(training_loss)

validation_loss = np.mean(history.history['val_loss'])
print(validation_loss)
