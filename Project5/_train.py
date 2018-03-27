import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

def train(car_features, notcar_features):
    X = np.vstack((car_features, notcar_features)).astype(np.float64)   
    scalar = StandardScaler().fit(X)
    scaled_X = scalar.transform(X)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=22)
    model = LinearSVC(loss='hinge') # Use a linear SVC 
    model.fit(X_train, y_train) # Train the classifier
    print('Test Accuracy of SVC = ', round(model.score(X_test, y_test), 4)) # Check the score of the SVC
    return model, scalar