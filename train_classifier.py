import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer




# !!! STEP 2: Train Model (This will take a few hours)
print("Start Reading data1.pickle File")
pick_in = open('data1.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()
print("Completed Reading data from data1.pickle File")

print("Randomly Shuffling the Data")
random.shuffle(data)
features = []
labels = []

totalDataSize = len(data)
currentIndex = 1
print("Start Extracting Features and Labels from Data")
for feature, label in data:
    print('Extracting Element from Data %s / %s' % (currentIndex, totalDataSize))
    features.append(feature)
    labels.append(label)
    currentIndex = currentIndex + 1
print('All Data Features and Labels Extracted Successfully')

print('Start Splitting Data into 70% Training, and 30% Testing')
xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.3, stratify=labels)
print('Obtained xtrain, xtest, ytrain, and ytest')

# print('Imputing NaN values with mean')
# # Impute NaN values with the mean
# imputer = SimpleImputer(strategy='mean')
# xtrain_imputed = imputer.fit_transform(xtrain)
# xtest_imputed = imputer.transform(xtest)
# print('Completed Imputing NaN Values with mean')

print("Creating SVC Model")
model = SVC(C=1,kernel='rbf',gamma='auto')
print("SVC Model Created")

print("Start Fitting Model using xtrain and ytrain")
# model.fit(xtrain_imputed, ytrain)
model.fit(xtrain, ytrain)
print('Model Fitting Completed')

print('Start Saving Model into model.sav File')
pick = open('model.sav', 'wb')
pickle.dump(model, pick)
pick.close()
print('Model Saved Successfully')

