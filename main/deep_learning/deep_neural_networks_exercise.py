# source: https://www.kaggle.com/code/johnfscott/exercise-deep-neural-networks/edit

# prevent floating-point rounding errors
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# import dependencies
import tensorflow as tf
import pandas as pd

# Setup plotting
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

# set path
path = "C:/Users/johnf/PycharmProjects/MachineLearning/input/"

concrete = pd.read_csv(path + 'concrete.csv')
# print(concrete.head())
# print(concrete.shape) # output: (1030, 9)

# setting 'CompressiveStrength' as target, input shape = [8]
# build model with 3 hidden layers
# each layer contains 512 units and ReLU activation
# include output layer of one unit with no activation

model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=512, activation='relu', input_shape=[8]),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=512, activation='relu'),
    # the linear output layer
    layers.Dense(units=1),
])

