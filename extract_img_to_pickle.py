import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# !!! STEP 1: Extract Image Data into Pickle Format
dir = 'D:\\Side_Projects\\emotion_classification\\datasets\\JAFFE'  # Modify this to the path where the training images are stored on your device.

categories = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


data = []


for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)
    
    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        emotionImg = cv2.imread(imgpath,0)
        
        try:
            # emotionImg = cv2.resize(emotionImg, (48,48))
            image = np.array(emotionImg).flatten()
            
            data.append([image, label])
            
        except Exception as e:
            pass
        


pick_in = open('data1.pickle', 'wb')
pickle.dump(data, pick_in)
pick_in.close()

