#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:30:21 2020

@author: Aron Djurhuus Jacobsen, Albert Kjøller Jacobsen & Phillip Chavarria Højbjerg
"""

#the main sorce of this code is from:
#Harrison Kinsley: Pythonprogramming.net, series link:  https://pythonprogramming.net/introduction-deep-learning-python-tensorflow-keras/




import numpy as np
import pandas as pd
import pickle


#the categories are defines
CATEGORIES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
#the pathway to the dataset is defined
filename = 'fer2013.csv'
#the dataset is imported and seperated by comma
data = pd.read_csv(filename, sep=",")

#Usage coloum is deleted from the dataset
data = data.drop('Usage',axis=1)


#this array contains the order of emotions in the pictures
emotion = np.array(data.emotion)

#variable 'pos' will contain the index value for all the pictures/emotions we'll use from the dataset
pos=[]  

#looping at the amount of categories, here its made sure to have 4000 photos for each emotion
for i in range(len(CATEGORIES)):
    #at the index of 1 we loop past the category 'Disgust' due to its low representation in the dataset
    if i == 1:
        i = i+1
    #the other categories
    else:
        #find the indexes of where the looping emotion is, use [0] because the it has 2 elements
        number = np.where(emotion==i)[0]
        #use only 4000 of these picture indexes for the looping emotion and extend it to 'pos'
        pos.extend(list(number[:4000]))


#variable 'X' will hold the pixel-pictures
X = []

#variable 'y' will hold the value for the emotions corresponding the the picture in 'X' 
y = []

#every picture in data represented as pixels is made to a numpy array
pixels = data.pixels.to_numpy()

#loop at the amount of pictures we'll use, to fill in variable 'X' and 'y'
for i in range(len(pos)):
    #find index for picture with 'pos' and extract it from pixels, then remove the space seperator,
    #and convert to a list stored in 'img_array'
    img_array = list(map(int, list(pixels[pos[i]].split(" "))))
    #find the image size for our squared ppictures, different for rectangles
    IMG_SIZE = int(np.sqrt(np.size(img_array)))
    #reshape the 1-D array to 2-D and add to 'X'
    X.append(np.reshape(img_array,(IMG_SIZE, IMG_SIZE)))
    #the values for the emotion corresponding to the picture is appended to 'y'
    if emotion[pos[i]] >= 2:
        #use this statements to adjust the values for emotions after 'Disgust' to their new value
        y.append(emotion[pos[i]]-1)
    else:
        y.append(emotion[pos[i]])
    
    
#Convert 'X' to a list 
X = list(X)
#reshape 'X' to contain (-1) as the amount of images, with the shape of images(2-D) and the value of their grayscale colour
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#convert 'y' to a numpy array
y = np.array(y)


#'pickle' is used to save the data and open it up in another python file.
#variable 'X' is saved
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

#variable 'y' is saved
pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()







