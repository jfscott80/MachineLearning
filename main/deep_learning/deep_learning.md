# Intro to Deep Learning

#### Using Keras and Tensorflow:
1. Create a **fully-connected** neural network architecture
2. Apply neural nets to two classic ML problems: **regression** and **classification**
3. Train neural nets with **stochastic gradient descent**
4. Improve performance with **dropout**, **batch normalization** and other techniques

## A Single Neuron
### Introduction
#### What is Deep Learning?
* **Deep learning** is an approach to machine learning characterized by deep stacks of computations.
* **Neural networks** have become the defining model; composed of neurons, where each neuron individually performs only a simple computation.
* The depth of computation allows for complexity of connections the neurons can form.

#### What are the most common applications of deep learning?
* Natural language translation
* Image recognition
* Game playing
* AI

### The Linear Unit
* Beginning with the fundamental component of a neural network: an individual neuron (or *unit*). 
$$y=wx+b$$  
* $x:$ the input. connects to the *weight*
* $w:$ the weight of the input's connection to the neuron. neural networks *learn* by modifying these weights
* $b:$ a special kind of weight called *bias*. this enables the neuron to moify the output independently of its inputs
* $y:$ the output of a neuron's activation
### Mutiple Inputs
* Expanding the neuron model:
$$y=x_0w_0+x_1w_1+x_2w_2+x_3w_3+...+x_nw_n+b$$
* A linear unit with two inputs will fit a plane.
### Linear Units in Keras
`keras.Sequential` creates a neural network as a stack of *layers*. Dense layering can create models to fit an *n dimensional* space.
```angular2html
from tensorflow import keras
from tensorflow.keras import layers

# Create a network with 1 linear unit
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])
# assign variables to weights
w, b = model.weights
```
* `units` define number of outputs
* `input_shape` define dimensionality of the array expected for each example in the training data. `[n]` will create a network accepting vectors of length `n`
* before the model is trained, each input is assigned a random weight and the bias is 0.0
* keras represents weights as tensors, but also uses tensors to represent data

---
## Deep Neural Networks
Add hidden layers to a network to uncover complex relationships. The hey idea is *modularity*, building up a complex network from simpler functional units. 
### Layers
Neural networks organize their neurons into layers. A dense layer is created by collecting together linear units with common inputs.  

* [**"Layers"**](https://www.tensorflow.org/api_docs/python/tf/keras/layers) is a general term to represent any kind of *data transformation*. 
    + **Convulutional** and **recurrent** layers transform data through the use of neurons and differ primarily in the pattern of connections they form
    + Others are used for feature engineering or just simple arithmetic

### The Activation Function
* This is the key step connecting dense layers into non-linear relationships.
* The most common acitvation function is the **rectifier function**: $max(0,x)$.
* Attaching the rectifier to a linear unit forms a **ReLU**: $max(0,w*x+b)$.
* Applying the the function to the ouput of a neuron will put a *bend* in the data.

### Stacking Dense Layers
* The layers before the output layer are sometimes called **hidden** since their is no direct observation of their outputs.
* The final *output* layer is a linear unit, making it appropriate to a regression task like predicting a numerical value.
* Other tasks like classification might require an activation function on the output.

### Building Sequential Models
* The `Sequential` model will connect tofether a list of layers in order from first to last.
* The first layer gets the input; the last layer produces the output.
```angular2html
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3, activation='relu'),
    # the linear output layer 
    layers.Dense(units=1),
])
```
* Pass all the layers together in a list like `[layer, layer, layer, ...]` instead of separate arguments
* To add an activation function to a layer, give its name in the `activation` argument.
```angular2html
layers,Dense(units=8, activation='relu')

# is equivalent to:

layers.Dense(units=8)
layers.Activation('relu')
```
### Alternatives to ReLU
`'relu'` activation belongs to family of variants including `'elu'`, `'selu'`, `'swish'`

---
## Stochastic Gradient Descent
#### Training a neural network
* At creation, neural networks don't *know* anything --their weights are set randomly.
* Neural networks *learn* to improve predictions of the target by adjusting the weights.
* In addition to training data, this requires two more things:
  1. A **loss function** that measures the network's prediction quality
  2. An **optimizer** to tell the network *how* to adjust the weights

### The Loss Function
* The **loss function** measures the disparity between the target's true value and the value the model predicts.
* Different problems require different loss functions; **regression** commonly uses **mean absolute error** (MAE)
$$MAE=(1/n)*\sum_{i=1}^{D}|y_i-\hat y|$$

* Besides MAE, other loss functions used in regression problems are mean-squared error and Huber loss.
* The model will use the loss function as a guide for finding the correct values of its weights; in other words, the loss function tells the network its objective.
#### Lower loss is better.

### The Optimizer - Stochastic Gradient Descent
* The optimizer is an algorithm that adjusts the weight to minimize the loss.
* Virtually all of the optimization algorithms used in deep learning belong to a family called **stochastic gradient descent**.
* They are iterative algorithms that train a network in *steps*.
* One step of training goes like this:
  1. Sample some training data and run it through the network to make predictions.
  2. Measure the loss between the predictions and the true values.
  3. Finally, adjust the weights in a direction that makes the loss smaller.

Each iteration's sample of training data is called a **minibatch** or **batch**. A complete round of the training data is called an **epoch**. The number of epochs you train for is how many times the network will see each training example.  

#### Learning Rate and Batch Size
Each iteration causes a shift in the line representing predictions in the direction of each batch. The size of these shifts are determined by the **learning rate**. Smaller learning rate means more minibatches are required for a network's weights converge to its best values.  
The learning rate and the size of the minibatches are the two parameters that have the largest effect on how the SGD training proceeds. Determining the right choices can be difficult; the interaction between these parameters is often subtle.  
Most work will not require extensive parameter searching. **Adam** is an SGD algorithm with an adaptive learning rate. Adam's "self-tuning" is a great general purpose optimizer.
#### Adding the Loss and Optimizer
Specifying the loss and optimizer can be done with a string:
```angular2html
model.compile(
    optimizer="adam",
    loss="mae",
)
```
The parameters can also be accessed directly through the [Keras API](https://keras.io/api/optimizers/).  

In ML context:
  * **gradient** refers to the direction and rate of change of a function with respect to its input parameters with regard to change in error
  * **descent** refers to moving in the direction of the negative gradient
  * the gradient of ML models with multiple parameters is a vector containing the partial derivatives of the cost function with respect to each parameter, indicating the direction of steepest descent in the parameter space
  * the higher the gradient, the faster the model can learn
  * the model stops learning if the gradient reaches zero
  * **stochastic** means "determined by chance"
  * the training is *stochastic* because the minibatches are random samples from the dataset  

---
## Overfitting and Underfitting
**Recall:** Keras keeps a history of the training and validation loss over the epochs during training. This section discusses how to intrepret these learning curves and use them to guide model development, finding evidence of *underfitting* and *overfitting*, and strategies for correction.
### Interpreting the Learning Curves
* The information in the training data comes in two kinds:
  1. **signal** is the part that generalizes, that helps our model make predictions from new data
  2. **noise** is the part that is *only* true of the training data; all of the random fluctuation coming from real-world data or all of the incidental, non-informative patterns that can't help the model make predictions
* Plotting the loss on the training set and adding the validation data creates **learning curves**.
* Validation loss gives an estimate of the expected error on unseen data.
* Training loss will go down either when:
  1. the model learns signal  $or$
  2. the model learns noise
* Validation loss will only go down when the model learns signal
* Therefore, the size of the **gap** between the validation curve and the training curve indicates precisely how much noise the model has learned.
#### The Trade-Off
* Because creating a model that learning all of the signal without any of the noise is highly improbable, we make a trade.
* We can get the model to learn more signal at the cost of learning more noise. As long as the trade is favorable, the validation loss will continue to decrease.
* At a certain point, the cost exceeds the benefit and validation loss begins to rise.
* **Underfitting**: the model hasn't learned enough signal
* **Overfitting**:the model has learned too much noise
### Capacity
A model's **capacity** refers to the size and complexity of the patterns it is able to learn, determined largely by the number of neurons and how they are connected.  
Underfitting may be manageable by increasing capacity either by making it:  
  * **wider** by adding units to existing layers $or$
  * **deeper** by adding more layers
```angular2html
model = keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(1),
])
# wider networks have an easier time learning more linear relationships
wider = keras.Sequential([
    layers.Dense(32, activation='relu'),
    layers.Dense(1),
])
# deeper networks prefer more non-linear relationships
deeper = keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1),
])
```
### Early Stopping
Keras includes early stopping through `class Callback`.  
A [callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks) is just a function that runs while the model trains. Early stopping will run after every epoch. [Defining new callbacks is simple](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LambdaCallback).
```angular2html
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
) # any number of callbacks may be passed to `fit`
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=500,
    callbacks=[early_stopping], # put your callbacks in a list
    verbose=0,  # turn off training log
)
```
---
## Dropout and Batch Normalization
There are dozens of kinds of [layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers) that may be added to a model. Some are like dense layers and define connections between neurons and others can do preprocessing or transformations of other sorts.  
### Dropout
Overfitting is caused by the network learning spurious patterns in the training data. To recognize these, the network will often rely on very specific combinations of weight; a kind of "conspiracy" of weights. Being so specific, they tend to be very fragile: remove one and the conspiracy falls apart.  

To break up these conspiracies **dropout** allows randomly dropping some fraction of a layer's inputs every step of training, making it harder for the network to learn these spurious patterns by forcing it to search for broad, general patterns whose weight patterns tend to be more robust.  

Dropout creates a kind of ensemble of networks. The predictions are no longer made by one big network, but instead by a committee of smaller networks.  
Individuals in the committee tend to make different kinds of mistakes, but be right at the same time, making the committee as a whole better than any individual.  
#### Adding Dropout
```angular2html
keras.Sequential([
    # ...
    layers.Dropout(rate=0.3), # apply 30% dropout to the next layer
    layers.Dense(16),
    # ...
])
```
### Batch Normalization
**Batch normalization** (or batchnorm) can help correct slow or unstable training.  
**Recall:** scaling data input falls under Neural Network Best Practices.  

Just as normalizing data before it goes into the network prevents unstable training behavior, so too does normalizing inside the network. As each batch comes in, the **batch normalization layer**:  
  * normalizes the batch with its own mean and standard deviation
  * puts the data on a new scale with two trainable rescaling parameters

#### Adding Batch Normalization
It can be added at almost any point in a network. After a layer:
```angular2html
layers.Dense(16, activation='relu'),
layers.BatchNormalization(),
```
between a layer and its activation function:
```angular2html
layers.Dense(16),
layers.BatchNormalization(),
layers.Activation('relu'),
```
Adding as the first layer, it can act as an adaptive preprocessor --a stand in for `sklearn.StandardScalar` (or other scalar).  

When adding **dropout**, you may need to increase the number of units in the `Dense` layers:
```angular2html
model = keras.Sequential([
    layers.Dense(1024, activation='relu', input_shape=[11]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1),
])
```
---
## Binary Classification
Classification into one of two classes is a common machine learning problem. Addressing classification problems with only two choices is binary, so we assign a **class label** to each: `category_A=0` and `category_B=1`.
### Accuracy and Cross-Entropy
**Accuracy** is the ratio of correct predictions to total predictions:  
`accuracy = number_correct / total` (perfect prediction equals `1.0`)  

Accuracy is a reasonable metric to use whenever the classes in the dataset occur with about the same frequency. The problem with most classification metrics, including accuracy, is that it can't be used as a loss function. **SGD** needs a loss function that changes smoothly; accuracy, being a ratio of counts, moves in "jumps". In other words, a discrete function cannot be a substitute for a continuous one. The substitute needed is the *cross-entropy* function.  

**Recall:** the loss function defines the *objective* of the network during training. With regression, the goal is to minimize the distance between the expected outcome and the predicted outcome. MAE measured this *distance*.  

Classification requires a measurement of the distance between one probability distribution to another;  **cross-entropy** penalizes incorrect probability predictions. Used for a classification loss, other metrics tend to improve along with it.
### Making Problems with the Sigmoid Function
The cross-entropy and accuracy functions both require probabilities as inputs $[0,1]$. To convert the real-valued outputs produced by a dense layer into probabilities, we use the **sigmoid function**.  

To get the final class prediction, we define the *threshold* probability. Typically this will be 0.5, so that rounding will give us the correct class:  

$[0,0.5)$ means `class=0`  |  $[0.5,1]$ means `class=1`

A 0.5 threshold is what Keras uses by default with its [accuracy metric](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/BinaryAccuracy).
```angular2html
model = keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=[33]),
    layers.Dense(4, activation='relu'),    
    layers.Dense(1, activation='sigmoid'), # included to produce class probabilities
])
```
Add the cross-entropy loss and accuracy metric in the `compile` method. For two-class problems, use the 'binary' versions. Adam works well with classification, so it can be used also.
```angular2html
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
) # early stopping callback included for convenience
early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=1000,
    callbacks=[early_stopping],
    verbose=0, # hide the output because we have so many epochs
)
history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5
history_df.loc[5:, ['loss', 'val_loss']].plot()
history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()

print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_binary_accuracy'].max()))
```