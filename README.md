# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # "Image References"
[image1]: ./output_images/Not-Car.png
[image2]: ./output_images/Not-Car_CH-1_HOG.png
[image3]: ./output_images/Not-Car_CH-2_HOG.png
[image4]: ./output_images/Not-Car_CH-3_HOG.png
[image5]: ./output_images/YCrCb_Detection_result.png
[image6]: ./output_images/YCrCb_Detection_result2.png
[image7]: ./output_images/Result_test1.jpg
[image8]: ./output_images/Result_test2.jpg
[image9]: ./output_images/Result_test3.jpg
[image10]: ./output_images/Result_test4.jpg
[image11]: ./output_images/Result_test5.jpg
[image12]: ./output_images/Result_test6.jpg
[video1]: ./project_video.mp4
[video2]: ./output_test_video.mp4
[video3]: ./output_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

My project includes the following files:
- src/myReport.py -- Main code script.
- src/lesson_functions.py -- Functions for the main code.
- output_*.mp4 -- Result videos.
- output_images/*.png -- Result pictures.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 8-44 of the file called `myReport.py`, and lines 44-64 of the file called `lesson_functions.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` (Lines 65-91 in `myReport.py`):

![alt text][image2]
![alt text][image3]
![alt text][image4]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found that the more the orientations are, the more accurate the result we shall get but with a slower speed. The pixels_per_cell and cells_per_block is the opposite. **So the parameters should be set considering both accuracy and the speed**. The code locates at the 53-62 lines in `myReport.py`.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using both color features and HOG features. The color features only contained the spatial features. I didn't use the histogram features cause the vehicles had various colors. I increased the data number by horizontal flip (Lines 14-27) and chose 20% of the total data to be the test data. The code locates at the 99-140 lines in `myReport.py`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code locates at the 206-222 lines in myReport.py.
I decided to search 5 different scale ranges over the image and came up with this:
![alt text][image5]
These ranges are set by 'y' values:

| Range  | y Value Range   |
|:--:|:-----|
| 1 | 400, 464   |
| 2 | 400, 528   |
| 3 | 400, 592   |
| 4 | 400, 656   |
| 5 | 400, 720   |

And the searching areas should look like this:
![alt text][image6]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on five scales using YCrCb 3-channel HOG features plus spatially binned color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image10]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_test_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
The code locates at the 339-373 lines in myReport.py.
I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. To enhance the detection stability, detections of last frame should also be considered. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video: `./output_test_video.mp4 `.
![alt text][video2]

### Here are six frames and their corresponding heatmaps, the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames, and the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]

---
### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

During my implementation of this project, the biggest problem is that the detection accuracy weren't always as high as the training results, and sometimes somewhat wobbly or unstable bounding boxes occured. Because vehicles have too many shapes and colors. Besides, the watching directions of the camera and the environments lights always change. When the vehicles went far or ran into shadows, misdetection occured. Although I augmented the data by flipping, the training data were still not enough. I think the convolutional neural network might hellp cause it uses the spatial information more.
