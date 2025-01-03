# from: https://www.kaggle.com/code/johnfscott/exercise-cross-validation/edit

# set path
path = "/input/"

import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
train_data = pd.read_csv(path + 'train.csv', index_col='Id')
test_data = pd.read_csv(path + 'test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice
train_data.drop(['SalePrice'], axis=1, inplace=True)

# Select numeric columns only
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
X = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()

print(X.head())
# So far, you've learned how to build pipelines with scikit-learn. For instance, the pipeline below will use SimpleImputer() to replace missing values in the data, before using RandomForestRegressor() to train a random forest model to make predictions. We set the number of trees in the random forest model with the n_estimators parameter, and setting random_state ensures reproducibility.

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))

from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("Average MAE score:", scores.mean())

# Step 1: Write a useful function
# create 3 folds. simple, cv=3

def get_score(n_estimators):
    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators, random_state=0))
    ])
    scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=3,
                                  scoring='neg_mean_absolute_error')
    return scores.mean()

# Step 2: Evaluate the model for eight different values: 50, 100, ...400
# each value represents the number of trees in the random forest

results = {}
for i in range(1,9):
    results[50*i] = get_score(50*i)

# visualization
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(list(results.keys()), list(results.values()))
plt.show()

# max MAE at n_estimators=100 <- worst
# min MAE at n_estimators=200 <- best

# https://en.wikipedia.org/wiki/Hyperparameter_optimization
# Grid Search