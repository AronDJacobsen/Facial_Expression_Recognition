#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:44:34 2020

@author: Aron Djurhuus Jacobsen, Albert Kjøller Jacobsen & Phillip Chavarria Højbjerg
"""


#the main sorce of this code is from:
#Harrison Kinsley: Pythonprogramming.net, series link:  https://pythonprogramming.net/introduction-deep-learning-python-tensorflow-keras/


import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pickle
import matplotlib.pyplot as plt

#from 'dataload' the 'X' and 'y' variables are opened and stored
#X
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
#y
pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

#Normalizing using scale because max=255 and min=0
X = X/255.0
#shuffling the 'X' and 'y' data together to make the emotions equally represented in training
X_shuffled, y_shuffled = shuffle(np.array(X), np.array(y))

#use the sequential model to construct layers
model = Sequential()


#using sequential the CNN will be designed
#in general the covolutional layers (CL) have a 3x3 filter, 'relu' as activation function, a 2x2 filter for max pooling layer, then and 50% dropout
#first is a CL with 256 nodes
model.add(Conv2D(256, (3, 3), input_shape=X_shuffled.shape[1:]))
#add the activation function
model.add(Activation('relu'))
#nex the maxpooling to downsize
model.add(MaxPooling2D(pool_size=(2, 2)))
#then dropout
model.add(Dropout(rate=0.5))

#next a CL with 128 nodes
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#then a CL with 64 nodes
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.5))

#flatten the array to 1-D before using the dense layers
model.add(Flatten())
#dense layers fo 32 nodes with 'relu' as activation function and dropout on 50%
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

#final layer of 6 nodes as the number of categories with softmax as activation
model.add(Dense(6))
#use softmax as works well with multi classification
model.add(Activation('softmax'))

#define loss, optimizer and metrics, 'sparse' to save time in memeory and computation same with 'adam' as optimizer and metrics to show performance
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
#define hvat to train on, as our data, batch size, epochs, and validation split
history = model.fit(X_shuffled, y_shuffled, batch_size=32, epochs=50, validation_split=0.1, shuffle=True)



#the sorce to this plotting code is inspired from:
#https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
#summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#the model is saved to the computer
model.save('CNNTrained.model')




