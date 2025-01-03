# train a model with gradient boosting

# set path
path = "C:/Users/johnf/PycharmProjects/MachineLearning/input/"

import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X = pd.read_csv(path + 'train.csv', index_col='Id')
X_test_full = pd.read_csv(path + 'test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and
                        X_train_full[cname].dtype == "object"]

# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)

# Step 1: Build model

# Part A
# Begin by setting my_model_1 to an XGBoost model. Use the XGBRegressor class, and set the random seed to 0 (random_state=0). Leave all other parameters as default.
# Then, fit the model to the training data in X_train and y_train.

from xgboost import XGBRegressor

# Define the model
model_1 = XGBRegressor(random_state=0)

# Fit the model
model_1.fit(X_train, y_train)

# Part B
# make predictions for the validation data
from sklearn.metrics import mean_absolute_error

predictions_1 = model_1.predict(X_valid)

# Part C
# calculate the model's MAE
mae_1 = mean_absolute_error(predictions_1, y_valid)
print("Mean Absolute Error 1: ", mae_1)


# Step 2: Improve the model
# build model 2. set XGBRegressor parameters to gain better results. fit the model.
# make predictions, calculate MAE

model_2 = XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=4, random_state=0)
model_2.fit(X_train, y_train,
             # early_stopping_rounds=5,
             eval_set=[(X_valid, y_valid)],
             verbose=False)

predictions_2 = model_2.predict(X_valid)
mae_2 = mean_absolute_error(predictions_2, y_valid)
print("Mean Absolute Error 2: ", mae_2)


# Step 3: Break the model
# again, but better
model_3 = XGBRegressor(n_estimators=200, learning_rate=0.05, n_jobs=4, random_state=0)
model_3.fit(X_train, y_train,
             # early_stopping_rounds=5, # this parameter is not recognized
             eval_set=[(X_valid, y_valid)],
             verbose=False)

predictions_3 = model_3.predict(X_valid)
mae_3 = mean_absolute_error(predictions_3, y_valid)
print("Mean Absolute Error 3: ", mae_3)

# Output:
# Mean Absolute Error 1:  18161.826171875
# Mean Absolute Error 2:  17101.578125
# Mean Absolute Error 3:  17035.58984375

# It is absolutely necessary for me to understand which is True
# [ ( larger numbers==better modeling ) | ( smaller MAE==better model ) ]
