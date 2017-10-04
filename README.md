# **Traffic Sign Recognition Program using TensorFlow**

The aim of this project was to accurately classify traffic signs from the [German Traffic Signs dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) by means of a convolutional neural network.

**Build a Traffic Sign Recognition Neural Network**

Here are the steps I followed:
* Load the dataset
* Explore, summarize and visualize the dataset
* Pre-process, balance and augment the dataset
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

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

### Data Set Summary
Stats are as follows:
* Number of training examples = 34799
* Number of vaidation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

### Visualisation of dataset
The German Traffic Signs dataset is visualised in the below histogram. It is split into 43 categories, with the number of images in each category on the y-axis. Note the extremely unequal distribution of the dataset. This can and will introduce bias into the training model, and will be addressed below. Also, a low of ~200 examples for some classes is simply too low, and will need to be augmented.

![alt_text][image1]

#### Data Preparation and Pre-processing


While the data can indeed simply be inserted into the model to begin training as-is, it would be to our benefit to process the data as follows:

##### 1. Grayscaling
Grayscaling reduces the image to a single layer instead of 3 RGB channels, drastically reducing the number of variables the network has to deal with. This results in vastly improved processing times. While loss of information due to loss of colour could be a concern, my tests with both RGB and BW images show no significant difference in performance.

##### 2. Equalisation
Equalisation of the image helps improve contrast and provides clearer, more well-defined edges. I initially used OpenCV's histogram equalisation, but found the results to be blurry and of poor contrast. skimage's adaptive CLAHE implementation took longer to process, but gave a far superior result.

##### 3. Normalisation
Normalisation involves scaling the image's intensity range from (0, 255) to (-1, 1). Smaller values with means about 0 prevent our gradients from going out of control and finding incorrect local minima.

##### 4. Augmentation (Transformation)
Due to the low number of examples from some classes, I've chosen to re-balance the dataset to prevent bias in the mode. There after, I tripled the size of the dataset over all classes, including the ones already heavily represented. I initially attempted penalised classification to make up for the dataset imbalance, but found good ol' data augmentation more effective.

##### 5. Shuffling
Rather self-explanatory - shuffles the dataset around so that the model doesn't train itself on the ORDER of the images instead of the features.

#### Visualisation of the Pre-processing Steps

![alt text][image3]

#### Grayscaling, Equalisation and Normalisation
My batch preprocess function is as follows. I began with conversion to grayscale, contrast limited adaptive histogram equalisation and normalisation from -1 to 1.

I initially used OpenCV's histogram equalisation, but found skimage's CLAHE implementation gave a much better result with more defined edges.
```python
from skimage import exposure
from sklearn.utils import shuffle
from skimage import exposure
import cv2

def batchPreprocess(X):
    X_norm = []
    for image in X:
            bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            equ = exposure.equalize_adapthist(bw)
            equ = (equ * 2.0/equ.max())
            equ = np.reshape(equ,(32,32,1))-1
            X_norm.append(equ)
    return np.array(X_norm)
```

#### Augmentation (Transformation)
Here, I employed rotation (within a certain degree range) and warping through projective transforms (skimage). Projective transforms were chosen due to their similarity to changes in camera perspective.

Credits to Alex for helping me figure out projective transforms! His code and write-up helped clarify the frankly rather confusing usage of skimage projective transforms. His blog here(https://navoshta.com/traffic-signs-classification/).

```python
from skimage.transform import ProjectiveTransform
from skimage.transform import rotate
from skimage.transform import warp

def randomTransform(image, intensity):
    
    # Rotate image within a set range, amplified by intensity of overall transform.
    rotation = 20 * intensity
    # print(image.shape)
    rotated = rotate(image, np.random.uniform(-rotation,rotation), mode = 'edge')
    
    # Projection transform on image, amplified by intensity.
    image_size = image.shape[0]
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

def batchAugment(X, y, multiplier = 2):
    X_train_aug = []
    y_train_aug = []
    for i in range(len(X)):
        for j in range(multiplier):
            augmented = randomTransform(X[i], 0.75)
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
Here's a histogram of the new distribution.

![alt text][image2]

Total size of 106226 images

```python
# Double the dataset size by rotation and transformation
X_train_aug, y_train_aug = batchAugment(X_train_aug, y_train_aug, 1)
y_train_aug = np.array(y_train_aug)

print("Augmentation complete!")
```
Output:
```
New augmented size is:  212452
Augmentation complete!
```
### Final Model Architecture
I modified the LeNet-5 architecture. Only major changes made to this model were the output shape (43 classes instead of 10) and the network parameters.

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
