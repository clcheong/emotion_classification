import os
import numpy as np
import cv2
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pywt

def extract_features(image_path, resize_shape=(128, 128)):
    # Resize image to ensure consistent feature dimensions
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
    img = cv2.resize(img, resize_shape)  # Resize image to a standard size
    
    coeffs2 = pywt.dwt2(img, 'haar')  # Perform 2D Discrete Wavelet Transform
    LL, (LH, HL, HH) = coeffs2  # Decompose the coefficients

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
