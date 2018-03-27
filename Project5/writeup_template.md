
# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/1.png
[image2]: ./output_images/2.png
[image3]: ./output_images/3.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is in the third cell of the ipynb file, and specifically, the `extract_features()` and `get_hog_features` are the method
 to implement this. The details of these 2 methods are just adopted from course contents.
#### 2. Explain how you settled on your final choice of HOG parameters.

For the color space, I chose `LUV` as it gave the best results in experiments. For the parameters, I chose `orient = 11`, `pix_per_cell = 16` and `cell_per_block = 2`. This is based on the setting of the course
 as well as several experiments.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM(in 6-7th cells, `train()`) with the default classifier parameters and only using HOG features. I got the accuracy of 98.23% on test set.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at several fixed scales all over the image, which is in `find_cars()` method in the 8th cell of the notebook. The details of this method is mainly adopted from the course contents.
In this method, I first find boxes in different scales based on the classifier we got before, after that, I applied heatmap and threshold to find the output boxes.



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Several results are showed below:

![alt text][image1]

![alt text][image2]

![alt text][image3]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I implemented heatmap from course content to reduce the overlapping detection, and set the threshold to be greater than 1 as a filter for false positive.
Besides that, I also do searching on different ranges for different scales, which eliminate some impossible detections, and help a lot for reducing the false positive detections.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problem during implementation is that there is still many detection that is not accuracy enough, also, the detection time is quite slow, as we are doing a lot of searching jobs, which prevents
 the algorithm from real-time. I did computer vision project before, using SSD(single shot detection) to do the fast object detection, so I think using neural network methods could make this much more robust and efficient.

