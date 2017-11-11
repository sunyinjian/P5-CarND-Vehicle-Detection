##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[hog_car_noncar]: ./output_images/hog_feature.png
[image_pipeline]: ./output_images/image_pipeline.png
[all_test_images]:./output_images/all_test_images.png
[video]: ./output_images/project_video_result.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 14 through 25 in the `training.py` 

I started by reading in all the `vehicle` and `non-vehicle` images.  

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `hog_channel=ALL`,  `pix_per_cell=8` and `cell_per_block=2`:

![alt text][hog_car_noncar]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and use the parameters to training a classifier. The parameters with the highest accuracy on test dataset were set as final choice .

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for training is in `training.py`. The `YCrCb` color space is used to generate HOG feature. The parameters used to generate HOG feature are `orientations=9`, `hog_channel=ALL`,  `pix_per_cell=8` and `cell_per_block=2`. The feature vector is scaled before training. Then a linear SVM classifier is used to train a classifier.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding windows were defined in lines 1 in [5] in `P5.ipynb`. The size and position of windows were set by the possible cars size and position in the image. Then, all the sliding windows with different settings were generated iterately in `find_all_cars` and `find_cars_using_single_seting` function in [4].

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The pipeline to detect cars is in `process_pipeline` function in [7]. Feature vector of each patch defined by sliding window is extracted using using hog sub-sampling and make predictions. If the hog features were suggested as a car, the windows would be saved and overlapped to generated new heatmap. More than one patch that are identified as car to avoid as much false positive detections as possible.

Here is one example of the pipeline.
![alt text][image_pipeline]

Here are the pipelines on all the test images.
![alt text][all_test_images]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project_video_result.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The filter used to avoid false positive detection is in `process_pipeline` function in [7]. I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. The pixels that has more than a certain number of windows identified as vehicles are set as final detection of vehicle. 


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline may fail where the hog feature is the same as the vehicle. The paremeters of sliding windows setting and the threshold of filter are sensitive to color, light or other factors. Deep learning may solve vehicle detection based on large training dataset.

