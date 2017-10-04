# **Traffic Sign Recognition Program using TensorFlow**

The aim of this project was to accurately classify traffic signs from the [German Traffic Signs dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) by means of a convolutional neural network.

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

You can view my workflow in Traffic_Signs_Classifier.ipynb.

[//]: # (Image References)
[image1]: ./images/origin_bar.png "Visualisation"
[image2]: ./images/updated_bar.png "New Visualisation"
[image3]: ./images/processed_img.png "Processed Images"
[image4]: ./images/all_types.png "43 Traffic signs"
[image5]: ./images/test_img.png "Web images"
[image6]: ./images/result1.png "Web Results 1"
[image7]: ./images/result2.png "Web Results 2"
[image8]: ./images/result3.png "Web Results 3"
[image9]: ./images/result4.png "Web Results 4"
[image10]: ./images/result5.png "Web Results 5"
[image11]: ./images/result6.png "Web Results 6"

### German Traffic Signs
![alt_text][image4]

### Data Set Summary & Exploration
Stats are as follows:
* No. of training images = 34799
* No. of vaidation images = 4410
* No. of testing images = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

### Visualisation of dataset
The German Traffic Signs dataset is visualised in the below histogram. There are 43 categories of traffic sign.

![alt_text][image1]

#### Dataset pre-processing

#### 1. Grayscaling
Convert RGB image to single layer grayscale image

#### 2. Equalisation
skimage's adaptive CLAHE implementation. It improve constrast and provide more detail edges.

#### 3. Normalisation
Normalisation scale pixel value from (0, 255) to (-1, 1).

#### 4. Augmentation (Transformation)
Images rotation

#### 5. Shuffling
Shuffles the dataset 

I use rotation (within a certain degree range) and warping through projective transforms (skimage).
Credits to Alex for sharing his code and ideas! His blog [here](http://navoshta.com/).

```python
from skimage.transform import ProjectiveTransform
from skimage.transform import rotate
from skimage.transform import warp

def randTransform(img, intensity):
    
    # Rotate image within a set range, amplified by intensity of overall transform.
    rotation = 20 * intensity
    # print(image.shape)
    rotated = rotate(img, np.random.uniform(-rotation,rotation), mode = 'edge')
    
    # Projection transform on image, amplified by intensity.
    image_size = img.shape[0]
    magnitude = image_size * 0.3 * intensity
    tl_top = np.random.uniform(-magnitude, magnitude)     # Top left corner, top margin
    tl_left = np.random.uniform(-magnitude, magnitude)    # Top left corner, left margin
    bl_bottom = np.random.uniform(-magnitude, magnitude)  # Bottom left corner, bottom margin
    bl_left = np.random.uniform(-magnitude, magnitude)    # Bottom left corner, left margin
    tr_top = np.random.uniform(-magnitude, magnitude)     # Top right corner, top margin
    tr_right = np.random.uniform(-magnitude, magnitude)   # Top right corner, right margin
    br_bottom = np.random.uniform(-magnitude, magnitude)  # Bottom right corner, bottom margin
    br_right = np.random.uniform(-magnitude, magnitude)   # Bottom right corner, right margin
    
    transform = ProjectiveTransform()
    transform.estimate(np.array((
            (tl_left, tl_top),
            (bl_left, image_size - bl_bottom),
            (image_size - br_right, image_size - br_bottom),
            (image_size - tr_right, tr_top))),
            np.array((
            (0, 0),
            (0, image_size),
            (image_size, image_size),
            (image_size, 0)
            )))
    transformed = warp(rotated, transform, output_shape = (image_size, image_size), order = 1, mode = 'edge')
    return transformed

def augmentDataset(X, y, multiplier = 2):
    X_train_aug = []
    y_train_aug = []
    for i in range(len(X)):
        for j in range(multiplier):
            augmented = randTransform(X[i], 0.75)
            X_train_aug.append(augmented)
            y_train_aug.append(y[i])
        X_train_aug.append(X[i])
        y_train_aug.append(y[i])
        
    X_train_aug, y_train_aug = shuffle(X_train_aug, y_train_aug)
    print("New augmented size is: ", len(X_train_aug))
    return X_train_aug, y_train_aug

def findIndexofSameClass(y, v):
    index = []
    for i in range(len(y)):
        if y[i] == v:
            index.append(i)
    return index
```

#### Image pre-processing steps

![alt text][image3]

### Final Model Architecture
I modified the LeNet-5 architecture. Dropout is implemented to improve the validation accuracy.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, output 28x28x6 	|
| RELU					|												|
| Dropout					|		0.9	rate								|
| Max pooling	      	| 2x2 stride,  output 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, output 10x10x16      									|
| RELU					|												|
| Dropout					|		0.8	rate								|
| Max pooling	      	| 2x2 stride,  output 5x5x16 				|
| Fully connected		| Input 400, output 120        									|
| RELU					|												|
| Dropout					|		0.7 rate										|
| Fully connected		| Input 120, output 84        									|
| RELU					|												|
| Dropout					|		0.5 rate										|
| Fully connected		| Input 84, output 43     									|
| Softmax				| With tf.reduce_mean loss and AdamOptimizer        									|

#### Parameters
Values were chosen after much trial and error, and observation of model behaviour. To train the model, the AdamOptimizer was used.
```python
rate = 0.0009
EPOCHS = 200
BATCH_SIZE = 128
```
#### Training Pipeline
```python
logits = traffic_signs_net(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```
#### Result
I manage to get **97.3%** accuracy on the validation set and **94%** accuracy on the test set.

#### Test Set Pulled From The Web

![alt text][image5]

Manage to get 6 out of 6 correct.
```
Test Accuracy = 1.000
```

Here are the results of the prediction:

![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]

### Conclusion
Althought the test accuracy is **94.0%**, it manage to predict correctly 6 out of 6 images pulled from the web. There is too many hyper paramaters to tune. It need to take a lot of time to train the data. It is a time consuming project and I learn a lot of this project.  
