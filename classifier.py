import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer


# !!! STEP 3: Test Trained Model (.sav file) - This will take 10-15 minutes
print('Reading data1.pickle')
pick_in = open('data1.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()
print('Read data1.pickle successfully')

print('Shuffling data')
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
    
print('Splitting data')
xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size = 0.3)
print('Data split successfully')

# print('Imputing NaN values with mean')
# # Impute NaN values with the mean
# imputer = SimpleImputer(strategy='mean')
# xtrain_imputed = imputer.fit_transform(xtrain)
# xtest_imputed = imputer.transform(xtest)
# print('Completed Imputing NaN Values with mean')

print('Loading Model')
pick = open('model.sav', 'rb')
model = pickle.load(pick)
pick.close()
print('Model loaded successfully')

print('Start Prediction')
prediction = model.predict(xtest)

accuracy = model.score(xtest,ytest)

categories = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

print('Accuracy: ', accuracy)

print(classification_report(ytest, prediction, target_names = categories))



# Calculate and display the confusion matrix
conf_matrix = confusion_matrix(ytest, prediction)
print("Confusion Matrix:")
print(conf_matrix)

# Display the confusion matrix as a heatmap
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(categories))
plt.xticks(tick_marks, categories, rotation=45)
plt.yticks(tick_marks, categories)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()



# print('Prediction is: ', categories[prediction[0]])

# emotionPic = xtest[0].reshape(48,48)

# plt.imshow(emotionPic, cmap='gray')
# plt.show()
