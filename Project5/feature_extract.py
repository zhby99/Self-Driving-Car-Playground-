import numpy as np
import cv2
import glob
import time
from skimage.feature import hog


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True: # Call with two outputs if vis==True to visualize the HOG
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:      # Otherwise call with one output
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# function to compute color histogram features 
def color_hist(img, nbins=32):
    ch1 = np.histogram(img[:,:,0], bins=nbins, range=(0, 256))[0]
    ch2 = np.histogram(img[:,:,1], bins=nbins, range=(0, 256))[0]
    ch3 = np.histogram(img[:,:,2], bins=nbins, range=(0, 256))[0]
    hist = np.hstack((ch1, ch2, ch3))
    return hist

# Define a function to extract features from a list of images
def img_features(feature_image,spatial_size, hist_bins, orient, pix_per_cell, cell_per_block):
    file_features = []
    # spatial feature
    spatial_features = cv2.resize(feature_image, spatial_size).ravel() 
    file_features.append(spatial_features)
    # color hist feature
    hist_features = color_hist(feature_image, nbins=hist_bins)
    file_features.append(hist_features)
    # hog feature
    hog_features = []
    for channel in range(feature_image.shape[2]):
        hog_features.append(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
    hog_features = np.ravel(hog_features)
    file_features.append(hog_features)
    return file_features

def extract_features(imgs, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for img in imgs:
        file_features = []
        image = cv2.imread(img)
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      
        file_features = img_features(feature_image, spatial_size,hist_bins, orient, pix_per_cell, cell_per_block)
        features.append(np.concatenate(file_features))
        feature_image=cv2.flip(feature_image,1)
        file_features = img_features(feature_image,spatial_size, hist_bins, orient, pix_per_cell, cell_per_block)
        features.append(np.concatenate(file_features))
        
        
    return features


def single_img_features(feature_image, spatial_size,hist_bins, orient, pix_per_cell, cell_per_block):     
    img_features = []
    spatial_features = cv2.resize(feature_image, spatial_size).ravel() 
    img_features.append(spatial_features)
    hist_features = color_hist(feature_image, nbins=hist_bins)
    img_features.append(hist_features)
    hog_features = []
    for channel in range(feature_image.shape[2]):
        hog_features.append(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
    hog_features = np.ravel(hog_features)
    img_features.append(hog_features)
    return img_features


def get_vehicle_list():
    images = glob.glob('*vehicles/*/*')
    cars = []
    notcars = []
    for image in images:
        if 'non' in image:
            notcars.append(image)
        else:
            cars.append(image)
    return cars, notcars