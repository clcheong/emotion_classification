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
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    coeffs2 = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs2

    # Function to calculate the statistical features for a sub-band
    def calc_stats(sub_band):
        sub_band = np.array(sub_band)
        mean = np.mean(sub_band)
        std = np.std(sub_band)
        median = np.median(sub_band)
        energy = np.sum(np.square(sub_band))
        skewness = skew(sub_band.flatten())
        kurt = kurtosis(sub_band.flatten())
        return [mean, std, median, energy, skewness, kurt]

    # Calculate features for each sub-band
    features_LL = calc_stats(LL)
    features_LH = calc_stats(LH)
    features_HL = calc_stats(HL)
    features_HH = calc_stats(HH)

    # Flatten and concatenate all features into a single vector
    features = features_LL + features_LH + features_HL + features_HH

    return features


def contains_any_substring(main_string, substrings):
    """
    Check if the main_string contains any of the substrings.

    Parameters:
    - main_string (str): The string to search within.
    - substrings (list of str): A list of substrings to search for.

    Returns:
    - bool: True if any substring is found in main_string, False otherwise.
    """
    for substring in substrings:
        if substring in main_string:
            return True  # Return True as soon as any substring is found
    return False  # Return False if no substrings are found


# !!! STEP 1: Extract Image Data into Pickle Format
dir = 'D:\\Side_Projects\\emotion_classification\\datasets\\task3_results'  # Modify this to the path where the training images are stored on your device.
categories = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
roi_folders = ['roi_mouth']  # Target Specific ROI
roi_filenames = ['_eyebrows', '_eyes', '_mouth']
data = []

for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)
    
    for imgFolder in os.listdir(path):
        imgFolderPath = os.path.join(path, imgFolder)
        
        for roiFolder in roi_folders:
            roiFolderPath = os.path.join(imgFolderPath, roiFolder)
                                

            for img in os.listdir(roiFolderPath):
                
                if contains_any_substring(img, roi_filenames) :
                
                    imgpath = os.path.join(roiFolderPath, img)
                    
                    try:
                        print("Extracting Features for: ", imgpath)
                        
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

