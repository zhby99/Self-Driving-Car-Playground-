# **Finding Lane Lines on the Road**


### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I removed the noises using GaussianBlur. After that, I applied Canny Edge detection from openCv to detect the edges. Then I defined a polygon to mask, and apply Hough on edge detected image to get the lines. Finally I draw the lines on the original image to get the result.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by collecting all the line points and apply polynomial fit on them. I removed all the points that are far from the line and finally get the 2 lines.


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the view of the car changes or the resolution of image changes. Now I am hardcoding the polygon region for detecting the edges. So when the view of the car changes or the resolution of image changes, it cannot get result good.

Another shortcoming could be curved street. I am using linear regression to get the 2 lines, so if the street is not straight, it will get wrong results.

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to use proportion to mask regions. So that we can still get good result when the resolution of image changes.
