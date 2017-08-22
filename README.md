**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1-1]: ./writeup_images/class_distributions_1.png "Initial Class Distribution"
[image1-2]: ./writeup_images/class_examples.png "Visualization 2"
[image2-1]: ./writeup_images/data_generation.png "Data Generation"
[image2-2]: ./writeup_images/generated_data.png "Generated Data"
[image2-3]: ./writeup_images/class_distributions_2.png "Class Distribution Increased"
[LeNet]: ./lenet.png "LeNet Architecture"
[Deep LeNet]: ./deep_lenet.png "Deep LeNet Architecture"
[image4-1]: ./writeup_images/mod_leNet_Gray_DO_0.6.png "Accuracy vs Loss"
[image5-1]: ./writeup_images/web_images.png "Web Images"
[image5-2]: ./writeup_images/softmax_predictions.png "Softmax Predictions"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You've come to the right place... Oh, and here is a link to my [project code.](https://github.com/todddangerfarr/sdcnd_p2_traffic_sign_classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy and pandas libraries to calculate summary statistics of the traffic signs data set:

* The size of training set is: 34799
* The size of the validation set is: 4410
* The size of test set is: 12630
* The shape of a traffic sign image is: 32 x 32 x 3
* The number of unique classes/labels in the data set is: 43

#### 2. Include an exploratory visualization of the dataset.

Initial data exploration and visualization is an important step in any data science, machine or deep learning task.  This provides a quick understanding of how the dataset is split, what it contains and if there's any class imbalance or unexpected problems.  For this dataset I wanted to look at the how many of each class the dataset contained and example images from each of these classes.

To visualize the class distribution I produced the following countplot using the Seaborn library:  

![Class Distribution][image1-1]

Looking at this distribution it makes perfect sense as this is probably about the relative frequencies that these sign types appear in the real world.  

In addition to the countplot, I wanted to understand what the images themselves actually looked like.  For this step I randomly selected 1 image from the training set for each class and plotted them in a grid with their associated names.  

![Image Samples][image2-1]

From this process it became obvious how different the lighting conditions and quality of each image was.  This is also intuitive as driving usually happens at all hours of the day (and night).   

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For preprocessing, I originally decided to keep the images as RGB because for some signs colors contain just as much information about it as the shape does.  However, this proved to be counter productive simply in terms of training time and ultimately in terms of model performance.  It was only after I applied a grayscale preprocessing step that my model started to really improve.  

In addition, as noticed in the data visualization and exploration stage of this project, the consistency for the lightness and contrast of the images is highly varied.  Therefore I evaluated two histogram normalization techniques.  The first is simply just normalizing the histogram of each channel across the 0-255 pixel range of that channel.  This will help increase the contrast of the image and bring each image closer to a normalized brightness.  The second normalization technique I tried is CLAHE, which stands for Contrast Limited Adaptive Histogram Equalization. For CLAHE the idea is similar but instead of normalizing the histograms across the entire image at once it will partition the image into small subsections and normalize the histograms locally to those sections.  This technique normalizes and boosts image contrast in a similar manner as histogram normalization but is less prone to highlight washout and crushed shadows.

In addition to preprocessing the training images, I decided to generate more data for the dataset by applying a set of transformations to each image for a given number of times.  Because CovNets naturally have built-in invariance to these transformations, this can be a great way to create more training images if the dataset is limited.  I used four random translations for each created image:

| Transformation         		| Limits	        					    |
|:-------------------------:|:-----------------------------:|
| Translation               | ± 2 pixels                    |
| Rotation                  | ± 15º                         |
| Warp                      | ± 2 pixels                    |
| Scale                     | 0.9 - 1.1 multiplier          |

Below is a quick verification of these created functions as well as a comparison between the two equalization techniques mentioned above.

![Preprocessing Visualization & Data Generation][image2-1]

From these images you can see each of the various preprocessing and generation functions applied to a set of three images pulled from the training data set.  While the color render is deeper on the normalized histogram image I decided to use the CLAHE method as this boosts contrast and image clarity while keeping an image color that more closely matches the original image.  

I then used these preprocessors and augmentation techniques to increase the overall size of the training set.  Iterating through the training set I added two augmented images for each image increasing the total training images to 104397.  However, I purposefully left the class imbalance as I felt that it was a good representation of the sign distribution that would be seen in a real world scenario.  My intuition here is that I'd actually want a model to always predict slightly biased towards signs that are more frequently occurring.  The updated class counts can be seen below:

![Increased Class Distribution][image2-3]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I started with a baseline of the LeNet Architecture, which is famous for classifying handwritten digits, on the unprocessed RGB color images.  

![LeNet Architecture][LeNet]

| Layer         		|     Description	        					          |
|:-----------------:|:-------------------------------------------:|
| Input         		| 32x32x3 RGB image   						          	|
| Convolution 5x5   | 1x1 stride, valid padding, outputs 28x28x6  |
| RELU					    |												                      |
| Max pooling	      | 2x2 stride,  outputs 14x14x6 			  	      |
| Convolution 5x5	  | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU					    |												                      |
| Max pooling	      | 2x2 stride,  outputs 5x5x16   				      |
| Flatten         	| 5*5*16 = 400        									      |
| Fully Connected	  | output = 120       									        |
| Fully Connected	  | output = 84       									        |
| Fully Connected	  | output = n_classes = 43     						    |
| Softmax           |                             						    |

From this baseline I added all my preprocessing steps, generated images, modified the LeNet to have deeper filter depths on the convolutions and added randomized dropout and regularization to keep the weights in check.  

![Deep LeNet w/Dropout Architecture][Deep LeNet]

| Layer         		|     Description	        					          |
|:-----------------:|:-------------------------------------------:|
| Input         		| 32x32x1 Grayscale Processed image   			 	|
| Convolution 5x5   | 1x1 stride, valid padding, outputs 28x28x16 |
| RELU					    |												                      |
| Max pooling	      | 2x2 stride,  outputs 14x14x16 				      |
| Convolution 5x5	  | 1x1 stride, valid padding, outputs 10x10x64 |
| RELU					    |												                      |
| Max pooling	      | 2x2 stride,  outputs 5x5x64   				      |
| Flatten         	| 5*5*64 = 1600        									      |
| Fully Connected	  | output = 240      									        |
| Dropout           | keep_prob = 0.6                             |
| Fully Connected	  | output = 120       									        |
| Dropout           | keep_prob = 0.6                             |
| Fully Connected	  | output = n_classes = 43     						    |
| Softmax           |                             						    |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

As previously mentioned the model was trained using the modified LeNet Architecture, what I've referred to as Deep LeNet with dropout of 0.6 on the fully connected layers.  I used an Adam Optimizer to minimize the loss function with L2 regularization applied to keep the weights in check.  L2 Regularization help penalize large weight which can directly correlate to model overfitting.  It is important to note that L2 Regularization also introduces another hyperparameter β.  The table below outlines the main hyperparameters and the values I used to train the model.

| Hyperparameter 		|     Value               					          |
|:-----------------:|:-------------------------------------------:|
| EPOCHS        		| 20    (limited by my CPU)                   |
| η (learning rate) | 0.001 (probably a little too large)         |
| µ (mu)		        |	0     (mean of initial weights)             |
| σ (sigma)         | 0.1   (standard deviation of initial weights|
| β (beta)      	  | 0.001 (L2 regularization penalty)           |
| Batch Size        | 128   (limited by memory)                   |

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The final Deep LeNet model results were:
* training set accuracy of 98.5%
* validation set accuracy of 97.8%
* test set accuracy of 97.0%

My approach was an iterative one.  I figured this would be the best way to really understand some of the concepts like dropout, L2 Regularization and how changing layer depths can effect the performance of the model.  Therefore, rather than hunting for a different model architecture, I started with the LeNet Architecture and slowly built a slightly more complex model that leveraged some other deep learning tricks to help improve performance.

Although a majority of the work was spent preprocessing the data and building additional augmented images for the training set, I did spend some time on the backend as well.  The first improvements were focused on regularization techniques.  I started by applying dropout on the fully connected layers.  Dropout forces the model to learn different and redundant representations of the activations because it can never rely on any of them being active during each pass.  I started with keep_prob of 0.5, meaning that half the activations were randomly destroyed between the fully connected layers.  However, I later increased this value to 0.6 which helped the learning rate by keeping more activations per iterations.  The second regularization technique I applied was an L2 regularization to the loss function which penalizes large, overfit weights.  I used the de facto value of 0.001 and it worked well in this regard.  However, despite regularization the biggest contributor to the model improvement was to increase the number of filters for deeper layers after each of the convolutions.  I empirically tested a few different values but ultimately settled on the aforementioned architecture.

Unfortunately I was limited significantly by the hardware that was available to me.  I chose to only run 20 EPOCHS with a higher learning rate (η) of 0.001 as the training time was just too long with a larger and smaller number, respectively.  Examining the Accuracy vs Loss curves, I feel this model could have continued to improve given more iterations and maybe a smaller learning rate to further prevent over shooting when attempting to find a global minimum.

![Accuracy vs Loss][image4-1]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I selected 8 additional traffic sign images from the web.  The originals as well as processed version of these images can be seen below:

![Web Images][image5-1]

Most of these images were pretty straight forward and had already been sized to 32x32 pixels.  However I purposefully picked a few additional large images of which I had to resize before preprocessing them.  I figured these would be the most difficult to classify as they may skew or lose information during the resizing operation.  

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Of the 8 signs from the web 2 where incorrectly classified resulting in a 75% test accuracy.

| Image			            |     Prediction	        					| PASS  |
|:---------------------:|:---------------------------------:|:-----:|
| Bumpy Road        		| Bicycles Crossing  							  | O     |
| General Caution   		| General Caution  									| X     |
| No Passing  					| No Passing    										| O     |
| Right of Way      		| Right of Way        			 				| O     |
| Road Work         		| Bumpy Road    			       				| X     |
| Roundabout        		| Roundabout          			 				| O     |
| Slippery Road      		| Slippery Road  			       				| O     |
| Stop              		| Stop          			       				| O     |

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Additionally beyond predictions alone I also included the top 5 softmax probabilities for each of the images.  These probabilities can be visualized in the graphs below.

![Softmax Predictions][image5-2]

From this visualization it becomes clear that the model is doing a very good job of classifying the images. Although it missed 2 out of the 8 images, the classes that were predicted are very similar in shape and content.  Also, the second most likely sign guesses for both of these images are the correct classes.

Overall, I'm really happy with my model and the fact that it is simple enough to be trained fairly quickly on a portable CPU while traveling overseas in remote factories!  I'd like to revisit this project in the future and look at more complex models as well as trying to better balance the classes to see if that would help with some of the incorrect predictions.  It was bumpy road and bicycles crossing, two of the less represented classes, that seemed to show up as incorrect predictions or cause problems.  With such small forward pass batches (again another hardware limitation) the model might not have had enough information in order to best learn how to distinguish between them.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Did not finish this section.  Will come back to it later...
