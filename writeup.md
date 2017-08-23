**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[random_pair]: ./images/random_pair.png
[hog_features]: ./images/hog_features.png
[pipeline_example]: ./images/pipline_example.png
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the `get_hog_features` function, located in the of the IPython notebook (`Vehicle Detection.ipynb`). This is a wrapper for the `hog()` function from the skimage package.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![random_pair]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I took random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. I used the `IPython.html.widgets` package to interactively change the paramters.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![hog_features]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. To have some idea how what is the influence of the `hog()` function paramters i used the `IPython.html.widgets` to play a bit with results. I ended up with several sets of parameters that looked promissing - the results for the car and non-car images looked diffrent to human's eye. But what was fine for a random pair of images not necessarily gave a good (recognition) result on the test images. I had to tweek the inital set of paramters to exclude many false positives or missing selections.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using spatial bining of color, histogram of color and histogram of oriented gradients features (see cell [22] in the python notebook). All three set of features were extracted from all three channels of the LUV color space. Before  training (cell [24]) the features were scaled using `StandardScaler` (cell [23]). The labeled images were split into the training and validation sets with the 80/20 ratio. I got the validation accuracy of 0.9879.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented the sliding window search in the `find_cars()` function. It computes the HOG on the whole image, and then slices it using the appropriate scale to extract the features.

I decided to use the following scales 1.25, 1.50 and 2.00. I treied smaller scales, like 0.5 or 1.0 too, but the computation time increased dramatically for them. To battle this problem I used python's coroutines (see the `get_car_boxes()` function), so the exectution of each scale could be done in parallel. This helped only partially. With the scale 0.5, the computation time per frame was close to a minute and it was very difficult to work effectively with. I am pretty sure the inclusion of the small scale would improve the result slightly.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![pipeline_example]

I tried to optimize the classifier using `GridSearchCV` from the sklearn package on the penalty paramter C in `LinearSVC`, but it always returned the lowest result for the penalty paramter from the initial list (1.0 in my case).
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  I also excluded the bounding boxes with a small size (2000 squared pixels) as they usually were coresponding to the false positives.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest problem in this project for me was the long computation time of the HOG features. I can imagine that going down with the image scale, to 0.5 for example, will improve the results. But I will be very hard to produce the real-time pipline with such a CPU hungry comptation. The other problem might be the diffrent weather or scenery that will cause many false positives or negatives.

