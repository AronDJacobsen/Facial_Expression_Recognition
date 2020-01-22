#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:52:15 2020

@author: Aron Djurhuus Jacobsen, Albert Kjøller Jacobsen & Phillip Chavarria Højbjerg
"""

#For our experiment why used the terminal to open the video and run this script at once;
#1. change directory to where this file and the experimental video is
#2. use the following commands: 
#open 'video name'.mp4;python livepredictor.py



import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import time


#Define the categories
CATEGORIES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


#create a function prepare its input frame
def prepare(filepath):
    #define the required image size
    IMG_SIZE = 48
    #use cv2 to resize the image
    new_array = cv2.resize(filepath, (IMG_SIZE, IMG_SIZE))
    #the image is returned in the required shape(format)
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#the trained model is imported
model = tf.keras.models.load_model("128-256-128-64-32-6-dropout50-50epochs.model")


#this code and other webcam code has its sorce from:
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
#display frame for 1 ms
key = cv2.waitKey(1)
#indicating we are using the webcam
webcam = cv2.VideoCapture(0)



#define the variables to contain 
angry = []
fear = []
happy = []
sad = []
surprise = []
neutral = []

#videofeeling is a short term container for average predicted emotions per 30 second emotion video
videofeeling = []
#total is the long term container for all the average predicted emotions per 30 second emotion video
total = []
#high contains the highest activated emotions per 30 second emotion video
#this variable is not used in the experiment, though used for instant results (not calibrated), more illutration in the end
high = []


#Face Detection using Haar Cascades
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
#import a classifier to detect faces
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

#use time to know when the camera started rolling
start = time.time()

#define variables before to make the loop start more smoothly
x, y, w, h = 0, 0, 48, 48

#this list will contain timestamps for each 'emotion' video with respect to 'time'
video = []
#looping from 1 to 12, to add the different timestamps
for i in range(1,13):
    #each video is 30 seconds hence timestamps set as such
    video.append(time.time() + i*30)

#create this loop to seperate the observation with respect to the video categories and know when to close the webcam/observations
for i in range(12):
    #create a while loop to let the webcam run
    while True:
        try:
            #use this to test if a video has finished and break out of the loop
            if time.time() > video[i]:
                break
            #if pressed q on the computer the webcam loop will close
            elif key == ord('q') & 0xff:
                print("Turning off camera.")
                webcam.release()
                cv2.destroyAllWindows()
                #due to bugs this loop is created
                for i in range (1,5):
                    cv2.waitKey(1)
                print("Camera off.")
                break
            
            #if nothing above, the webcam keeps rolling as with observations
            else:
                #capturing frame by frame
                check, frame = webcam.read()
                key = cv2.waitKey(1)
                #make the frames grayscale as in the trained pictures
                grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #use face cascade to detect the persons face more close up as in the trained pictures
                faces = faceCascade.detectMultiScale(grayFrame,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(200, 200),
                    flags=cv2.CASCADE_SCALE_IMAGE)
        
                # Draw a rectangle around the faces, extracting the coordinates observed from 'faces'
                for x, y, w, h in faces:
                    rectangle = cv2.rectangle(frame, (x, y), (x+w, y+h), (90, 50, 255), 2)
                
                #using the coordinates from 'faces', ROI extracts the values inside the rectangle (rectangle on image, ROI)
                ROI = grayFrame[y:y+h, x:x+w]
                
                #define ROI as 'float32' due to error if not
                ROI = ROI.astype('float32') 
                #ROI goes first through the 'prepare' function,
                #then uses predict to get a list from imported trained model of current emotional state
                prediction = model.predict([prepare(ROI)])
                
                #pos will contain the index value of the most activated emotion from the model
                pos = np.where(prediction==np.max(prediction))[1][0]
                #text will contain the written emotion
                text = CATEGORIES[pos]
                
                #now the list for each emotion gets added with the current observed value    
                angry.append(prediction[0][0])
                fear.append(prediction[0][1])
                happy.append(prediction[0][2])
                sad.append(prediction[0][3])
                surprise.append(prediction[0][4])
                neutral.append(prediction[0][5])
                
                #***
                #not used in the experiment, though useful so see how the model is predicting live
                #first the text display setting are set
                #font = cv2.FONT_HERSHEY_SIMPLEX
                #scale = 1
                #color = (90, 50, 255)
                #thickness = cv2.FILLED
                #cv2.putText(frame, text, (200, 200), font, 1, color, thickness=2)
                #then the webcam will open
                #cv2.imshow("Emotionalligent", frame)              
                #***
        
        #if the program is interruptet, these statements will run and the loop will break     
        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            cv2.destroyAllWindows()
            for i in range (1,5):
                cv2.waitKey(1)
            print("Camera off.")
            print("Program ended.")
            break
     
    #find the sum of how many times emotions have been observed (amount of loops)
    sumFeeling = np.size(angry)
    #find the average value for each emotions in these 30 observed seconds
    angryP = np.sum(angry)/sumFeeling
    fearP = np.sum(fear)/sumFeeling
    happyP = np.sum(happy)/sumFeeling
    sadP = np.sum(sad)/sumFeeling
    surpriseP = np.sum(surprise)/sumFeeling
    neutralP = np.sum(neutral)/sumFeeling
    
    #a list of the average prediction value in these 30 seconds for each emotion is appended
    videofeeling.append([round(angryP,5), round(fearP,5), round(happyP,5), round(sadP,5), round(surpriseP,5), round(neutralP,5)])
    
    #the containers for all emotions observed for the 30 sec. video are reset
    angry = []
    fear = []
    happy = []
    sad = []
    surprise = []
    neutral = []
    
    #'high' gets the most activated emotions in these 30 sec. in words
    high.append(CATEGORIES[np.where(videofeeling==np.max(videofeeling))[1][0]])
    
    #total gets the list with the average activation value per 30 sec. video
    total.append(videofeeling)
    #'videofeeling' gets reset
    videofeeling = []
    


#after the loop, the matrix 'total' is printed, copied, and pasted in the next scripts for plots
#it is very important to save this value, an alternative could be to use 'pickle'
print(total)


#below is the code for instant results and plots from the expriment
#keep in mind, the results are not calibrated


#high is now printed to show dominating emotion per video
print(high)


#out of the 5 different videos, insert current video used in experiment
vid = 'insert video numnber (not in text)'
#'high' and the printed statement below can now be compared
if vid == 1:
    #Video 1
    print(['Happy','Surprise','Fear','Neutral','Angry','Fear','Happy','Surprise','Neutral','Sad','Angry','Sad'])
    order = ['Happy','Surprise','Fear','Neutral','Angry','Fear','Happy','Surprise','Neutral','Sad','Angry','Sad']
if vid == 2:
    #Video 2
    print(['Neutral','Surprise','Surprise','Fear','Neutral','Happy','Angry','Happy','Sad','Sad','Fear','Angry'])
    order = ['Neutral','Surprise','Surprise','Fear','Neutral','Happy','Angry','Happy','Sad','Sad','Fear','Angry']
if vid == 3:
    #Video 3
    print(['Sad','Happy','Surprise','Sad','Fear','Angry','Neutral','Fear','Surprise','Neutral','Happy','Angry'])
    order = ['Sad','Happy','Surprise','Sad','Fear','Angry','Neutral','Fear','Surprise','Neutral','Happy','Angry']
if vid == 4:
    #Video 4
    print(['Surprise','Angry','Surprise','Neutral','Neutral','Fear','Sad','Happy','Fear','Happy','Sad','Angry'])
    order = ['Surprise','Angry','Surprise','Neutral','Neutral','Fear','Sad','Happy','Fear','Happy','Sad','Angry']
if vid == 5:
    #Video 5
    print(['Sad','Neutral','Surprise','Fear','Neutral','Sad','Angry','Surprise','Happy','Happy','Angry','Fear'])
    order = ['Sad','Neutral','Surprise','Fear','Neutral','Sad','Angry','Surprise','Happy','Happy','Angry','Fear']


#the different average values for emotions in each video is split up
for i in range(12):
    angry.append(total[i][0][0])
    fear.append(total[i][0][1])
    happy.append(total[i][0][2])
    sad.append(total[i][0][3])
    surprise.append(total[i][0][4])
    neutral.append(total[i][0][5])


#code below to plot is inspired from: 
#https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html


#now the plot will be printet
#define amount of subplots
fig, axs = plt.subplots(6, sharex=True)
#give the plot the title 'Exp. 1'
fig.suptitle('Exp. 1')
#define the values
axs[0].plot(np.arange(12), angry, '.', linestyle='-')
#set the title for the subplot
axs[0].set_title('Angry')

#same procedure...
axs[1].plot(np.arange(12), fear, '.',linestyle='-')
axs[1].set_title('Fear')

axs[2].plot(np.arange(12), happy, '.',linestyle='-')
axs[2].set_title('Happy')

axs[3].plot(np.arange(12), sad, '.',linestyle='-')
axs[3].set_title('Sad')

axs[4].plot(np.arange(12), surprise, '.',linestyle='-')
axs[4].set_title('Surprise')

axs[5].plot(np.arange(12), neutral, '.',linestyle='-')
axs[5].set_title('Neutral')

#setting the x label
for ax in axs.flat:
    ax.set(xlabel='Expected emotion per 30 sec. video')
    plt.xticks(np.arange(12), order, rotation=45)

#hide x labels for top plots
for ax in axs.flat:
    ax.label_outer()

#setting height between plots
plt.subplots_adjust(hspace=2)

#setting the y label to be in the middle and not too close to the y values
fig.text(0.06, 0.5, 'Calibrated observation\n\n\n\n\n\n', ha='center', va='center', rotation='vertical')
plt.show()






