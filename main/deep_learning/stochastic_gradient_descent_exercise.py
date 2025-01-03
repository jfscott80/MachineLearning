"""In the Fuel Economy dataset your task is to predict the fuel economy of an automobile"""
"""given features like its type of engine or the year it was made."""

# Setup plotting
import matplotlib.pyplot as plt
# from learntools.deep_learning_intro.dltools import animate_sgd
plt.style.use('seaborn-v0_8-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

# load dataset, preprocess
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split

# set path
path = "C:/Users/johnf/PycharmProjects/MachineLearning/input/"
fuel = pd.read_csv(path + 'fuel.csv')

X = fuel.copy()
# Remove target
y = X.pop('FE')

# Note: parameter 'sparse' was renamed to 'sparse_ouput' and has FutureWarning attached to it currently.
# Extra note: PyPi doesn't yet support scikit-learn 0.16.1
preprocessor = make_column_transformer(
       (StandardScaler(),
       make_column_selector(dtype_include=np.number)),
       # (OneHotEncoder(sparse=False),
       (OneHotEncoder(),
       make_column_selector(dtype_include=object)),
)

X = preprocessor.fit_transform(X)
y = np.log(y) # log transform target instead of standardizing

input_shape = [X.shape[1]]
print("Input shape: {}".format(input_shape)) # Output: Input shape: [50]

# target is 'FE' and the remaining columns are the features
print(fuel.head())
print(pd.DataFrame(X[:10, :]).head())

# define the network
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1),
])

# before training the network, define the loss and optimizer using compile. use Adam and MAE
model.compile(
       optimizer='adam',
       loss='mae',
)

history = model.fit(
       X, y,
       batch_size=128,
       epochs=200,
)

# convert the training history to a dataframe, plot
history_df = pd.DataFrame(history.history)
history_df.loc[5:, ['loss']].plot();

# if the learning curves haven't leveled off, increasing epochs may decrease loss --improving the model

# learning rate and batch size control:
# how long it takes to train the model
# how noisy the learning curves are
# how small the loss becomes

# Epoch 195 - 200 loss
# 195: 0.0291
# 196: 0.0277
# 197: 0.0243
# 198: 0.0310
# 199: 0.0343
# 200: 0.0278

'''
smaller batch sizes gave noisier weight updates and loss curves. This is because each batch is a small sample of data and smaller samples tend to give noisier estimates. Smaller batches can have an "averaging" effect though which can be beneficial.

Smaller learning rates make the updates smaller and the training takes longer to converge. Large learning rates can speed up training, but don't "settle in" to a minimum as well. When the learning rate is too large, the training can fail completely. (Try setting the learning rate to a large value like 0.99 to see this.)
'''