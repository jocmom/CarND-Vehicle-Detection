## Vehicle Detection - Project 5 of Udacity Self-Driving Car Nanodegree
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Following files are provided:

* P5.ipynb - the main jupyter notebook
* box.py - used to store the bounding boxes information
* feature_extraction.py - defines function for HOG and color space feature extractions
* multi_ploy.py - helper functions to plot multiple images
* README.md - here you are
* several test and output images
* project_video_out.mp4 - output video
* svm_Ycrcb.p - pickle file with the trained SVM

[//]: # (Image References)
[video1]: ./project_video_output.mp4
[car_notcar]: ./output_images/car_notcar.png
[notcar]: ./output_image/notcar.png
[normalized]: ./output_images/normalized.png
[hog_car]: ./output_images/hog_car.png
[hog_notcar]: ./output_images/hog_notcar.png
[unnormalized]: ./output_images/unnormalized.png
[all_windows]: ./output_images/all_windows.png
[box_final]: ./output_images/box_final.png
[heatmap]: ./output_images/heatmap.png
[final_heatmap]: ./output_images/final_heatmap.png
[labels]: ./output_images/labels.png
[final]: ./output_images/final.png

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook.  

In the second cell of the jupyter notebook `P5.ipynb` all `vehicle` and `non-vehicle` images are loaded. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][car_notcar]

In cell 3 of the notebook we use the functions from `feature_extraction.py` to explore the HOG features. By applying the HOG operation the image is divided in smaller cells and the magnitude of gradients in these cells is calculated and all of these gradients will be used as features. 

Additionally to the HOG features we can use color features to dectect cars. I tested different color spaces like "RGB", "HLS", "LUV", "Ycrcb", etc.    
I combined the the color channels with the `skimage.hog()` which is used in the function `get_hog_features` (feature_extraction.py: line 15) and tried different parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). 
Below is an example of the hog features from the "car" and a "not car" classes using the R channel from the `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][hog_car]

![alt text][hog_notcar]

#### 2. Explain how you settled on your final choice of HOG parameters.

In cell 4 of the jupyter notebook I use following parameters/features for the classifier:
- color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
- orient = 9  # HOG orientations
- pix_per_cell = 8 # HOG pixels per cell
- cell_per_block = 2 # HOG cells per block
- hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
- spatial_size = (16, 16) # Spatial binning dimensions
- hist_bins = 32    # Number of histogram bins

I played around with all of the features. Using more than 9 orientations was no significant improvement, same for `cell_per_block` and `pix_per_cell`. The `spatial_size` (16,16) was the best compromise regarding speed and classification performance. Compared to the HOG features the color features are a little bit underrepresented therefore I increased the `hist_bins` parameter to 32. in my tests the LUV, YUV and Ycrcb color spaces with all channels achieved the best results. I decided to use the Ycrcb space because it has a little bit less false positives.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

A linear SVM is trained in cell 5 of the notebook. I used the provided images from the course and used the same amount of "car" and "not car" images. The `StandardScaler` scikit-learn library normalizes the data and the `train_test_split` function shuffle and split the data in 80% training and 20% test data. With the described parameters we got a test accuracy of 99%. Here you can see an example of unnormalized and normalized data:

![alt text][unnormalized]

![alt text][normalized]

In cell 6 there is possibility to save and load a trained SVM.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

From the test images you can see that the street starts around 400+ in y direction and cars are passing till 600. Therefore the search area in the y direction is restricted to this area. There is no restriction in x direction.
For the search I use quadratic windows with sizes from 128x128 in the bottom of this area to 64x64 in the top. In the example below the function `slide_window` (feature_extraction.py: line 146) provided by the course is used to display these windows. When I used smaller windows also in the bottom I got more false positives on the lanes. The overlap of 75% got much better results 0% or 50% overlap.

![alt text][all_windows]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on nine scales using all channels of the YCrCb HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Below are two examples of the bounding boxes in red and the final box in blue. 

To optimize the performance the `find_cars` (feature_extraction.py: line 249) is called which only extracts HOG features once for the desired area. On my laptop it takes around 1.2 seconds per frame. Also restricting the area improved this a lot.

![alt text][box_final]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used the class `Box` in the file box.py to record the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and add this heatmap to queue of heatmaps (box.py: line 26)from the last 5 frames to average the heatmap (line 27). Then I thresholded the average map to identify vehicle positions (line 29).  
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap (line 32).  I then assumed each blob corresponded to a vehicle. In the method `find_final_boxes` starting on line 68 I constructed bounding boxes to cover the area of each blob detected.  

### Here is an example of the final output and the corresponding heatmap:
![alt text][final_heatmap]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][labels]

### Here the resulting bounding boxes are drawn onto one frame in the series:
![alt text][final]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* I had some issues to apply the right tresholds to get rid of the false positives but still detecting the white car during the turn. Using the 'Ycrcb' helped and also averaging the heatmap but maybe I focued too much on this and generalization isn't so good anymore.

* Processing of each frame isn't so fast - using a neural network should improve performance on real time a lot

* In the end there was one car detected on the opposite lane. Here it is not a false positive but this should be handled differently. By tracking the direction with a Kalman Filter we would distinguish approaching cars. 

* Right now we are using only one color space maybe we can combine color spaces or some channels of these spaces

## References
[Udacity Repository](https://github.com/udacity/CarND-Vehicle-Detection)

http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html