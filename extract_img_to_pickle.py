import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from scipy.stats import kurtosis, skew



def extract_features(image_path):
    img = cv2.imread(image_path)
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Median
    median_intensity = np.median(img)

    # Standard Deviation
    std_dev_intensity = np.std(img)

    # Mean Absolute Deviation (MAD)
    # mad_intensity = np.mean(np.abs(img - np.mean(img)))

    # Median Absolute Deviation (MAD)
    # mad_median_intensity = np.median(np.abs(img - np.median(img)))

    # Kurtosis
    kurtosis_intensity = kurtosis(img.flatten())

    # Skewness
    skewness_intensity = skew(img.flatten())

    # Moment (let's consider the 3rd moment - skewness again)
    # moment_intensity = np.mean(np.power((img - np.mean(img)), 3))

    # Variance
    # variance_intensity = np.var(img)

    # Mean
    mean_intensity = np.mean(img)

    # Covariance (using the same variable for illustration)
    covariance_intensity = np.cov(img.flatten(), img.flatten())[0, 1]

    # Energy
    energy_intensity = np.sum(np.square(img))

    return [
        median_intensity,
        std_dev_intensity,
        # mad_intensity,
        # mad_median_intensity,
        kurtosis_intensity,
        skewness_intensity,
        # moment_intensity,
        # variance_intensity,
        mean_intensity,
        covariance_intensity,
        energy_intensity
    ]



# !!! STEP 1: Extract Image Data into Pickle Format
dir = 'D:\\Side_Projects\\emotion_classification\\datasets\\raw'  # Modify this to the path where the training images are stored on your device.
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

