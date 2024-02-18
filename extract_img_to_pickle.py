import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from scipy.stats import kurtosis, skew
import pywt



def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
    coeffs2 = pywt.dwt2(img, 'haar')  # Perform 2D Discrete Wavelet Transform
    LL, (LH, HL, HH) = coeffs2  # Decompose the coefficients

    # You can further process these coefficients or use them directly as features.
    # For example, you can calculate statistics like mean, variance, etc. for each sub-band.

    # Flatten the coefficients and create a feature vector
    features = np.hstack([np.array(LL).flatten(), np.array(LH).flatten(), 
                          np.array(HL).flatten(), np.array(HH).flatten()])

    return features


# !!! STEP 1: Extract Image Data into Pickle Format
dir = 'D:\\Side_Projects\\emotion_classification\\datasets\\JAFFE'  # Modify this to the path where the training images are stored on your device.
categories = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
data = []

for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)
    
    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        
        try:
            # Extract features using the function
            features = extract_features(imgpath)

            # Append features and label to the data list
            data.append([features, label])

        except Exception as e:
            pass

# Save the data to a pickle file
pick_in = open('data1.pickle', 'wb')
pickle.dump(data, pick_in)
pick_in.close()

