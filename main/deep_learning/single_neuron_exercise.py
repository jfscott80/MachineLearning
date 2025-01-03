# Setup plotting
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid') # matplotlib shipped seaborn styles are updated
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

import pandas as pd
# set path
path = "C:/Users/johnf/PycharmProjects/MachineLearning/input/"
red_wine = pd.read_csv(path + 'red-wine.csv')
print(red_wine.head())
# print(red_wine.shape) # (1599, 12)

# setting 'quality' as target and remaining columns as features
# input_shape = [11]

# define a linear model
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(units=1, input_shape=[11])
])
w, b = model.weights    # model.get_weights() is also an option
# print("Weights\n{}\n\nBias\n{}".format(w, b))
# untrained, each input has it's own random weight and bias 0.0

