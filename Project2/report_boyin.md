# **Traffic Sign Recognition**
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/vis1.png "Visualization1"
[image2]: ./examples/vis2.png "Visualization2"
[image4]: ./examples/1.jpeg "Traffic Sign 1"
[image5]: ./examples/2.jpeg "Traffic Sign 2"
[image6]: ./examples/3.jpeg "Traffic Sign 3"
[image7]: ./examples/4.jpeg "Traffic Sign 4"
[image8]: ./examples/5.jpeg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/zhby99/Self-Driving-Car-Playground-/blob/master/Project2/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a circular graph shows the distribution of first 10 labels in training set and test set.

![alt text][image1]
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For the pre-processing, I only did the normalization with (x-128 / 128) for all pixel. In my opinion, color is also an important element we use to identify traffic signs, which is not like handwriting digits, where color is not useful, so I decided to not convert the image to grayscale.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		    |     Description	        					                    |
|:---------------------:|:-----------------------------------------------------:|
| Input         		    | 32x32x3 RGB image   						            	        |
| Convolution 5x5x12   	| 1x1 stride, valid padding, outputs 28x28x12 	        |
| RELU					        |											                                 	|
| Max pooling	        	| 2x2 stride,  outputs 14x14x12 				                |
| Convolution 5x5x32	  | 1x1 stride, valid padding, outputs 10x10x32           |
| RELU					        |											                         	        |
| Max pooling	        	| 2x2 stride,  outputs 5x5x32 				                  |
| Fully connected		    | input: 800, output: 240, with dropout, keep rate = 0.9|
| Fully connected		    | input: 240, output: 100, with dropout, keep rate = 0.9|
| Fully connected		    | input: 100, output: 43      			                    |
| Softmax				        |        								                	              |




#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer which is same with LeNet, also I used the batch size of 128, with 25 epochs and learning rate to be 0.0005

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998  
* validation set accuracy of 0.944
* test set accuracy of 0.928

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
Just LeNet.
* What were some problems with the initial architecture?
It got an accuracy about 0.87, which does not meet our desires. Also, the accuracy on training set is much higher than that of valid set, so I think it is under overfitting.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I slightly changed the number of kernels for ConvNets as I think traffic signs are more complex than digits, so need to find more features. Also I added dropout to reduce overfitting.
* Which parameters were tuned? How were they adjusted and why?
Number of epochs was increased while learning rate was decreased in order to make the learning more smoothy.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
I think the number of kernels and the dropout rate discussed above are important design choices.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

None of these images should be difficult to classify as I picked the images in which the traffic signs occupy most of the area.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Turn right ahead    		| Turn right ahead  									|
|Road work   			|Road work							|
| Speed limit (30km/h)				| Speed limit (30km/h)										|
| Priority road	      		| Priority road				 				|
| Stop			| Priority road  							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is lower than the accuracy on test set. As the images I found from web maybe not as standard as the test set images.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located at the bottom of the Ipython notebook.
For all these 5 images, the model gives a probability higher than 0.95 for its final prediction, and only gives a very small number for other options.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
