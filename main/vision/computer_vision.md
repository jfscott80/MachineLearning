---
title: "Computer Vision"
author: "John Scott"
description: <a href="https://www.kaggle.com/learn/computer-vision">Computer Vision</a>
---
# Computer Vision
#### Objectives:
* Build an **image classifier** with Keras deep-learning networks
* Design **custom convnets** with reusable blocks
* Explore the fundamental ideas behind visual **feature extraction**
* Boost models with the art of **transfer learning**
* Utilize **data augmentation** to extend datasets

#### Introduction
How does a neural network *understand* a natural image well enough to solve the same problems that human visual systems can solve?  

The neural networks that are best at this task are **convolutional neural networks** (also called **convnet** or **CNN**). Convolution is the mathematical operation applied to a network's layers to provide their unique structure. 
#### The Problem of Image Classification
Given a picture, can a computer be trained to identify what it is a picture of?
#### Advanced Applications
Beyond this lesson, lies **generative adversarial networks** and **image segmentation**

---
## The Convolutional Classifier
A convnet used for image classification consists of two parts:  
   * **convolutional base**: extracts the features from an image, formed primarily of layers performing the convolution operation, and often include other layers as well
   * **dense head**: determines the class of an image, formed primarily of dense layers, but might include other layers like dropout.
#### What is meant by "visual feature"?
A feature can be a line, a color, a shape, a pattern, or any combination thereof.
### Training the Classifier
The goal of the network during training is to learn:  
* which features to extract from the image (base)
* which class goes with what features (head)  

Generally, convnets are rarely made from scratch. Instead, we reuse the base of a pretrained model and attach an untrained head to it. Essentially, we have a starting model that already understands how to extract features and train it to classify with the new layers.  
Very accurate classifiers can be created from relatively little data, because the head only consists of a few dense layers. Reusing a pretrained model is the **transfer learning** technique.
#### Steps
1. Load data
2. Define pretrained base
   * the most commonly used dataset for pretraining is [ImageNet](https://image-net.org/about.php)
   * the keras [API Module](https://www.tensorflow.org/api_docs/python/tf/keras/applications) includes a variety of models pretrained on ImageNet
3. Attach head
4. Train
5. Evaluate
---
## Convolution and ReLU
### Feature Extraction
The **feature extraction** performed by the base consists of three basic operations:
1. **Filter** an image for a particular feature (convolution)
2. **Detect** that feature within the filtered image (ReLU)
3. **Condense** the mage to enhance the features (maximum pooling)

Typically, the network will perform several extractions in parallel on a single image. It's not uncommon for the final layer in the base to be producing over 1000 unique visual features.
### Filter With Convolution
```angular2html
model = keras.Sequential([
    layers.Conv2D(filters=64, kernel_size=3), # activation is None
    # More layers follow
])
```
To understand these parameters, examine their relationship to the weights and activations of the layer.
#### Weights
The weights a convnet learns are referred to as **kernels**. Here, they will be represented as small arrays. A kernel operates by scanning over an image and producing a *weighted sum* of pixel values. In this way, a kernel will act sort of like a polarized lens, emphasizing or de-emphasizing certain patterns of information.  

**Kernels** define how a convolutional layer is connected to the layer that follows. The kernel above will connect each neuron in the output to nine neurons in the input. `kernel_size` sets the dimensions, telling the convnet how to form the connections. Usually, a kernel will have odd-numbered dimensions like $(3,3)$ or $(5,5)$, leaving a single pixel sat in the center; this is not required.
#### Activations
Activations in the network are called **feature maps**; the result from a filter applied to an image, containing the visual features that the kernel extracts.  
Generally, what a convolution accentuates in its inputs will match the shape of the *positive* numbers in the kernel.  
The `filters` parameter tells the convolutional layer how many feature maps to create as output.
### Detect With ReLU
**Recall:**
* **ReLU:** $f(x)=max(0,x)$
* a neuron with a rectifier attached is called a *rectified linear unit*
* the rectifier function zeroes out any negative values
* activation functions are **non-linear**
```angular2html
model = keras.Sequential([
    layers.Conv2D(filters=64, kernel_size=3, activation='relu')
    # More layers follow
])
```
#### Example: Define a kernel as "edge detection"
```angular2html
import tensorflow as tf

kernel = tf.constant([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
])
# view kernel
plt.figure(figsize=(3, 3))
show_kernel(kernel)
```
#### More kernel examples:
```angular2html
# Bottom Sobel
kernel = tf.constant([
    [-1, -2, -1],
    [0,   0,  0],
    [1,   2,  1],
])
# Emboss
kernel = tf.constant([
    [-2, -1,  0],
    [-1,  1,  1],
    [ 0,  1,  2],
])
# Sharpen
kernel = tf.constant([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0],
])
```
#### Note:
When defining a kernel, remember that the sum of the numbers in the kernel determine how *bright* the final image is. General practice holds this sum within the interval $(0,1)$, though it is not required.

---
## Maximum Pooling
A convnet performs feature extraction in three steps. We covered how the first two steps occur in a `Conv2D` layer with `relu` activation.  
The final operation of this sequence: **condense** with **maximum pooling** occurs in a `Maxpool2D` layer.  
### Condense with Maximum Pooling
```angular2html
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(filters=64, kernel_size=3), # activation is None
    layers.MaxPool2D(pool_size=2),
    # More layers follow
])
```
A `Maxpool2D` layer is much like a `Conv2D` layer, except that it uses a simple maximum function instead of a kernel.  
* The parameter `pool_size` $\approx$ `kernel_size`
* `Maxpool2D` doesn't have trainable weights

The point of this final step is to collect dead space in the feature map generated during **Detect**. Carrying all the zero activations throughout the entire network increases the size of the model without adding much useful information. So we **condense** the feature map to only retain the most useful part: the feature itself.  

**Max pooling** take a patch of activations in the original feature map and replaces them with the maximum activation in that patch; when applied after ReLU activation, pooling increases the proportion of active pixels to zero pixels, "intensifying" features.  

Each application of max pooling **does** reduce the size of the matrix $\log_2 M_{size}$
### Translation Invariance
The zero pixels still carry one piece of important information: *positional*. When `MaxPool2D` removes some of these pixels, it removes some of the positional information in the feature map.  
This gives a convnet a property called **translation invariance**: a convnet with maximum pooling tends not to distinguish features by their *location* in the image.  
#### Recall:
* **translation** is the mathematical term for change of position in a matrix without rotating it, changing its shape, or its size (not to be confused with *transformation*).  

Max pooling only creates translation invariance in a network over *small distances*. Features that begin far apart will remain distinct after pooling; only *some* of the positional information was lost, but not *all*.

This invariance to small differences in the position of features isn't seen as a bug; it's a nice property for an image classifier to have.  
Just because of differences in perspective of framing, the same kind of feature might be positioned in various parts of the original image, but we would still like for the classifier to recognize that they are the same.  
Because this invariance is *built-into* the network, we can use much less data for training because we *don't* have to train the network to ignore that difference.

This gives convolutional networks an efficiency advantage over a network with only dense layers.

#### Key Problems with CNNs:
* **Limited invariance with large translations:** large object shifts may cause the network to mis-classify objects significantly displaced in the image  
* **Dependence on training data distribution:** if training data contains objects primarily centered in the image, the network may struggle with objects positioned off-center
* **Challenges with fine-grained recognition tasks:** in cases like facial recognition, even small translations can significantly impact classification accuracy
* **Impact on object detection and segmentation**
#### Potential Solutions
* **Data augmentation:** [Survey of Modern Approaches](https://www.sciencedirect.com/science/article/pii/S2590005622000911), [Neural Radiance Fields (NeRF)](https://en.wikipedia.org/wiki/Neural_radiance_field)
* **Advanced pooling strategies:** [Spatial Transformer Networks](https://proceedings.neurips.cc/paper_files/paper/2015/file/33ceb07bf4eeb3da587e268d663aba1a-Paper.pdf), [PyTorch Implementation of STNs](https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html)
* **Designing translation-equivalent architectures**
* **Attention mechanisms:** focus on relevant parts of the image regardless of position

Max pooling has largely superseded **average pooling** within the convolutional base. There is, however, a kind of average pooling still widely used in the *head* of a convnet: **global average pooling**.  
#### Global Average Pooling
A `GlobalAvgPool2D` layer may be used as an alternative to some or all of the hidden `Dense` layers in the network:
```angular2html
model = keras.Sequential([
    pretrained_base,
    layers.GlobalAvgPool2D(),
    layers.Dense(1, activation='sigmoid'),
])
```
Instead of using an additional `Flatten` layer to "unstack" the feature and transform the 2D data to 1D data needed, the `GlobalAvgPool2D` layer achieves the transformation by replacing the entire feature map with its average value. Destructive, but effective in some cases.
* one **5 X 5** feature map is reduced to a single value, making the number of parameters needed to represent these features by a factor of **25**

Global average pooling is often used in modern convnets. Instead of trying to map *raw* features to classes, it allows for a singular calculation:
* the convnet designates a single high-level feature (like a wheel or a window)
* pooling a map gives us a single number --large if the feature is present, small if absent
---
## The Sliding Window
#### Recall:
The three operations that carry out feature extraction from an image are:
1. *filter* with a **convolution** layer
2. *detect* with **ReLU** activation
3. *condense* with a **maximum pooling** layer  

The convolution and pooling layers are both performed over a **sliding window**. With convolution, this window is given by the dimensions of the kernel, the parameter `kernel_size`. Pooling gives a *pooling window* with the parameter `pool_size`.  
There are two additional parameters affecting both layers:
* `strides` of the window
* whether to use `padding` at the image sides
### Stride
The distance the window moves at each step is called the **stride**.  
`strides=(2, 1)` means "move from left to right two pixels, move from top to bottom one pixel".  

* Convolutional layers wil most often have `strides=1` (if the value of the strides is the same number for both directions, this syntax will suffice) because we want high-quality features to use for classification.  
* Increasing the stride means we will miss out on potentially valuable information in our summary.  
* Max pooling layers will almost always have stride values greater than one, but not larger than the window itself.
###  Padding
During sliding window computations, the boundaries present another challenge. On the one hand, the window must remain entirely within the input image; on the other, the window should sit squarely over each pixel examined.  
The padding parameter determines how the boundary pixels are handled. Tensorflow provides two choices:
* `padding='valid`: the convolution window remains entirely inside the input, with the drawback being:
  * the output shrinks (loses pixels) and shrinks more for larger kernels, limiting the number of layers the network can contain
* `padding=same`: **pad** the input with just enough zeroes around its borders to make the size of the output the *same* as the input
  * this can dilute the influence of pixels at the borders
* most modern convnets use some combination of these two

*Stacking* convolutional layers can increase the effective window size.  
Convolution can be applied to one-dimensional data like **time series**.
#### The Receptive Field
Trace back all the connections from some neuron and eventually you reach the input image. All the pixels that a neuron is connected to is that neuron's **receptive field**. The receptive field identifies which parts of input image a neuron receives information from.  

If the first layer is a convolutional layer with **3 X 3** kernels, then each neuron in that layer gets input from a 3 X 3 patch of pixels (except maybe at the border).

---
## Custom Convnets
### Simple to Refined
A single round of of feature extraction can only extract relatively simple features: a line, contrasts. These are too simple to solve most classification problems. Instead, convnets will repeat this extraction over and over, so that the features become more complex and refined as they travel deeper into the network.
### Convolutional Blocks
It does this by passing them through long chains of **convolutional blocks** which perform this extraction.  
These convolutional blocks are stacks of `Conv2D` and `MaxPool2D` layers. Each block represents a round of extraction and composing them allows the convnet to combine and recombine the features produced. Deep structures of convnets achieves sophisticated feature engineering and superior performance.
---
## Data Augmentation
### The Usefulness of Fake Data
The best method for improving convnet performance is to train it on more data. More data lets the model learn to generalize better.  
One way to get more data is to use the data you already have. If the images can be transformed while preserving the class, the classifier learns to ignore those kinds of transformations.  
Ex: when training to identify vehicles, the model should recognize a car whether it is facing left or facing right.  

**Augment** the training data with flipped images, the classifier learns that left or right is a difference it can ignore.
### Using Data Augmentation
Data augmentation is usually done *online* as the images are being fed into the network for training.  

#### There are many types of transformations:
* **Geometric Transformations:**
  * rotation: rotating the image by different angles
  * scaling: resizing the image by changing its dimensions
  * flipping: flipping the image horizontally or vertically
  * cropping: cutting out a portion of the image
  * translation: moving the image horizontally or vertically
  * shearing: slanting the image along an axis
* **Color Adjustments**
  * brightness: changing the overall brightness of the image
  * contrast: adjusting the difference between light and dark areas
  * saturation: modifying the intensity of colors
  * hue: shifting the color spectrum
  * color jitter: randomly altering multiple color properties
* **Noise Addition:**
  * Gaussian noise: adding random numbers to the image
  * salt and pepper noise: randomly setting pixels to black or white
* **Blurring:**
  * applying Gaussian blur or other blurring techniques

Each time an image is used during training, a new random transformation is applied; in this manner, the model is always seeing something just a little bit differently, adding variance that improves the model's performance against new data.  
It's important to ensure that the *type* of transformations being used do not mix up the classes. A simple example of this kind of mis-application:
* training a network to identify digits
* applying rotation
* sixes and nines become indistinguishable
#### More Datasets
* [MNIST](https://yann.lecun.com/exdb/mnist/)**:** tens of thousands of images of handwritten digits
* [EuroSAT](https://zenodo.org/records/7711810#.ZAm3k-zMKEA)**:** land use and land cover classification dataset
* [Tensorflow](https://www.tensorflow.org/datasets)**:** an extremely large collection of datasets available