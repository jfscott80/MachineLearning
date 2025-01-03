# data source: https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-01-21
"""This exercise demonstrates early stopping callback implementation to prevent overfitting"""

# prevent floating-point rounding errors
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Setup plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks

# set path, read in data
path = "C:/Users/johnf/PycharmProjects/MachineLearning/input/"
spotify = pd.read_csv(path + 'spotify.csv')

"""Task: Predict the popularity of a song based on various audio features."""

X = spotify.copy().dropna()
y = X.pop('track_popularity')
artists = X['track_artist']

features_num = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms']
features_cat = ['playlist_genre']

preprocessor = make_column_transformer(
    (StandardScaler(), features_num),
    (OneHotEncoder(), features_cat),
)

# We'll do a "grouped" split to keep all of an artist's songs in one
# split or the other. This is to help prevent signal leakage.
def group_split(X, y, group, train_size=0.75):
    splitter = GroupShuffleSplit(train_size=train_size)
    train, test = next(splitter.split(X, y, groups=group))
    return (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])

X_train, X_valid, y_train, y_valid = group_split(X, y, artists)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
y_train = y_train / 100 # popularity is on a scale 0-100, so this rescales to 0-1.
y_valid = y_valid / 100

input_shape = [X_train.shape[1]]
# print("Input shape: {}".format(input_shape)) # Output: [18]

"""Start with a simple linear model network (low capacity)."""
# model = keras.Sequential([
#     layers.Dense(1, input_shape=input_shape), # "When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead."
# ])
# model.compile(
#     optimizer='adam',
#     loss='mae',
# )
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_valid, y_valid),
#     batch_size=512,
#     epochs=50,
#     verbose=0, # suppress output since we'll plot the curves
# )
# history_df = pd.DataFrame(history.history)
# history_df.loc[0:, ['loss', 'val_loss']].plot()
# # print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min())); # Output: 0.2002
#
# # it's not uncommon for the curves to follow a 'hockey stick' pattern, making the final part of the training hard to
# # see.
# # Start the plot at epoch 10
# history_df.loc[10:, ['loss', 'val_loss']].plot()
# # print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min())); # Output: 0.1979

# Commented out previous model to prevent warning messages
# new model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model.compile(
    optimizer='adam',
    loss='mae',
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
    verbose=0, # suppress output since we'll plot the curves
)
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
# print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min())); # Output: 0.1989

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=5, # how many epochs to wait before stopping
    restore_best_weights=True,
)
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model.compile(
    optimizer='adam',
    loss='mae',
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50, # stopped at epoch 11/50
    callbacks=[early_stopping]
)
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min())); # Output: 0.1943 found at epoch 6/50









