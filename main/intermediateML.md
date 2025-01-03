# Python Machine Learning II

I. Handling missing values, categorical variables  
II. Design pipelines  
III. Cross-Validation  
IV. Leakage and other common mistakes

sample random forest models
```angular2html
# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)
from sklearn.ensemble import RandomForestRegressor

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]
```
---
```angular2html
from sklearn.metrics import mean_absolute_error

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))
```
---
## I. Handling Missing Values and Categorical Variables
### Missing Values
Most machine learning libraries (including sci-kit) give an error running a model that uses missing values.  
#### Evaluating Data Set
Does the data set contain missing values?
* Find number of rows, number of columns using `shape()`.
* Identify columns with missing values and number missing in column.
```angular2html
# Shape of training data (num_rows, num_columns)
print(X_train.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
```
#### Simple option: Drop columns with missing values
Unless most of the data in the column is missing, this option eliminates the use of potentially useful data.  
```angular2html
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
```
#### Better option: Imputation
**Imputation** fills in the missing values with some number. This can be the mean, median or any other number. This substitution will not be exact, but it usually leads to more accurate modeling.  
```angular2html
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
```
#### An Extension to Imputation
In addition to the standard imputation, generate another column of boolean values that identifies the adjusted rows.  
```angular2html
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))
```
---
### Categorical Variables
Read in of sample data set containing numerical and categorical values:
```angular2html
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X = pd.read_csv('../input/train.csv', index_col='Id') 
X_test = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll drop columns with missing values
cols_with_missing = [col for col in X.columns if X[col].isnull().any()] 
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
    train_size=0.8, test_size=0.2, random_state=0)

X_train.head()
```
Categorical data must be modified before training a model.  
#### Drop columns with categorical data
```angular2html
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
```
Before ordinal encoding, investigate unique data remaining. Specifically in the 'Condition2' column.
```angular2html
print("Unique values in 'Condition2' column in training data:",
        X_train['Condition2'].unique())
print("\nUnique values in 'Condition2' column in validation data:",
        X_valid['Condition2'].unique())
```
#### Ordinal encoding
class sklearn.preprocessing.OrdinalEncoder(*, categories='auto', dtype=<class 'numpy.float64'>, handle_unknown='error', unknown_value=None, encoded_missing_value=nan, min_frequency=None, max_categories=None)  

This involves creating a function to fit an ordinal encoder to the training data then use it to transform both the training and validation data.  

Fitting an ordinal encoder to a column in the training data creates a corresponding integer-valued label for each unique value that appears in the training data. In the case that the validation data contains values that don't also appear in the training data, the encoder will throw an error, because these values won't have an integer assigned to them.  

The simplest approach is to drop the problematic categorical columns.

---
Identify problematic columns:  
```angular2html
# Categorical columns in the training data
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that can be safely ordinal encoded
good_label_cols = [col for col in object_cols if 
                   set(X_valid[col]).issubset(set(X_train[col]))]
        
# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))
        
print('Categorical columns that will be ordinal encoded:', good_label_cols)
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)
```
Then drop the bad columns and use ordinal encoding on the good columns.
```angular2html
from sklearn.preprocessing import OrdinalEncoder

# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

# Apply ordinal encoder 
ordinal_encoder = OrdinalEncoder()
label_X_train[good_label_cols] = ordinal_encoder.fit_transform(X_train[good_label_cols])
label_X_valid[good_label_cols] = ordinal_encoder.transform(X_valid[good_label_cols])

print("MAE from Approach 2 (Ordinal Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
```

#### One-Hot Encoding
class sklearn.preprocessing.OneHotEncoder(*, categories='auto', drop=None, sparse_output=True, dtype=<class 'numpy.float64'>, handle_unknown='error', min_frequency=None, max_categories=None, feature_name_combiner='concat')  

Identify categorical columns by name and total number of unique values within:
```angular2html
# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])
```
One-hot encoding adds columns to the dataset.  
[ ( number of rows * number unique variables ) - number of rows ]

As an example, consider a dataset with 10,000 rows, and containing one categorical column with 100 unique entries. The number of entries added to this dataset with one-hot encoding is $=1*10^4 * 100 - 1*10^4$. Ordinal encoding would add zero new entries.  

One-hot encoding categorical columns with cardinality $\lt 10$:
```angular2html
# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

from sklearn.preprocessing import OneHotEncoder

# Note: parameter 'sparse' was renamed to 'sparse_ouput' and has FutureWarning attached to it currently.
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Ensure all columns have string type
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)
```
---
## II. Pipelines
Organizing complex models and data preprocessing is essential for scalability. Pipelines are a method of bundling many steps into one easy to compile step.  

#### Step 1: Define Preprocessing Steps
  
class sklearn.compose.ColumnTransformer(transformers, *, remainder='drop', sparse_threshold=0.3, n_jobs=None, transformer_weights=None, verbose=False, verbose_feature_names_out=True, force_int_remainder_cols=True)  

Similar to how pipeline bundles together preprocessing and modeling steps, we use the ColumnTransfer class to bundle together different preprocessing steps. The next code block:  
+ imputes missing values in *numerical* values
+ imputes missing values and applies a one-hot encoding to *categorical* data
```angular2html
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
```
#### Step 2: Define the model
```angular2html
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)
```
#### Step 3: Create and Evaluate the Pipeline
* The pipeline preprocesses the training data and fit the model in a single line of code, instead of doing the imputation, one-hot encoding, and model training in separate steps.
* The unprocessed validation features are supplied to the `predict()` command and the pipeline automatically preprocesses the features before generating predictions.
```angular2html
from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
```
## III. Cross-Validation
A better way to test models.  
In general, the larger the validation set => less noise => better measure of model quality. This is a trade-off, since increasing the size of the validation set requires decreasing the size of the training data => worse models.
#### What is cross-validation?
Running the modeling process on different subsets of the data to get multiple measures of model quality. A simple version of this idea is to partition the full dataset into five subsets and perform five experiments. Each experiment holds back one subset of the data as validation. This is referred to as **folding**.  
#### When should cross-validation be used?
This process creates a more accurate measure of model quality which becomes more important as more modeling decisions are made. Running time will increase with number of folds performed linearly.  
Small datasets have low computational costs; cross-validation should always be used.  
The threshold for cost is subjective. If the decision is made to use cross-validation, pipelines are essential.  
```angular2html
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50,
                                                              random_state=0))
                             ])
from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates Negative MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)
```
sklearn.model_selection.cross_val_score(estimator, X, y=None, *, groups=None, scoring=None, cv=None, n_jobs=None, verbose=0, params=None, pre_dispatch='2*n_jobs', error_score=nan)  
+ The above use of the *negative* MAE scoring function is almost unheard of outside of this implementation. Scikit-learn has a convention where all the metrics are defined so that higher numbers are better.  
+ Since we want a single measure of model quality to compare alternative models. So we take the average across experiments.  
```angular2html
print("Average MAE score (across experiments):")
print(scores.mean())
```
## XG Boost
Gradient boosting is the most accurate modeling technique for structured data. This is another **ensemble method**, like RandomForest modeling.  
Begin with a single, naive model and iteratively cycle to improve the ensemble.  
This uses gradient descent on the loss function of each model's predictions to determine the parameters used in the new model. Reduction of loss is paramount; the loss function used is subjective.  
```angular2html
# eXtreme Gradient Boosting => XGB
from xgboost import XGBRegressor # import the scikit-learn API for XGBoost

my_model = XGBRegressor()
my_model.fit(X_train, y_train)

# make predictions, evaluate the model
from sklearn.metrics import mean_absolute_error

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
```
### Parameter Tuning
`n_estimators`  
* How many times through the modeling cycle / equal number of models included in the ensemble.
* Too *low* a value causes underfitting, which leads to inaccurate predictions on both testing and training data. 
* Too *high* a value causes overfitting, causing accurate predictions on training data at the cost of accuracy against testing data.
* Typical values range from 100-1000, but this depends on `learning_rate`
```angular2html
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train)
```
`early_stopping_rounds`
* This value represents the number of cycles **in a row** that have no improvement in the validation score and stop the iteration process early
* Common practice is to set `n_estimators` to a high value and use this to find the optimal time to stop iterating
* This also requires setting aside some data for calculating the validation scores.
* Best-case: use this parameter to find the optimal number of models, then rerun the model with the found value set
```angular2html
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)],
             verbose=False)
```
`learning_rate`
* Instead of getting predictions by simply adding up the predictions from each component model, we can multiply each new model value by a small number before adding it.
* Now, each tree added to the ensemble helps us less.
* Increasing `n_estimators` even higher is less likely to cause overfitting.
* In general, the idea is to set the number of cycles very high, prevent overfitting with learning rate diminishing returns, and let evaluations determine when to stop early.
```angular2html
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
```
`n_jobs`
* On larger datasets where runtime is a consideration, using parellelism allows fast model building. 
* This will be set to number of cores on the machine used
* Doesn't help smaller datasets
* Won't improve the model
```angular2html
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
```
---

## Data Leakage
### What is it, Why is it bad, How does it happen
**Data Leakage** happens when training data contains information about the target, but similar data will not be available when the model is used for prediction. This leads to high performance on the training set (and possibly also on the validation set), but the model will perform poorly in production.   

**Leakage** occurs in two ways:
+ **target leakage** 
+ **train-test contamination**

#### Target leakage
Occurs when the model predictors include data that will not be available at the time predictions are made. This is a matter of *timing* or *chronological order* that data becomes available, not merely whether a feature helps make good predictions.  
The example given involves predicting pneumonia cases. Using data from a column "took antibiotics" will create better models, but is absolutely ridiculous to include in the model because it contains a value directly changed when the target value is realized. Any variable updated at or after the Moment of Prediction **must** be excluded to prevent target leakage.  
#### Train-test contamination
Occurs when training data is not separated from validation data during preprocessing. Pipelines are a common practice to avoid this problem.

### Detecting and Removing Target Leakage
The example used represents credit card acceptance prediction based on the following features:  
```angular2html
y = 'card'
features = [ 'reports', 'age', 'income', 'share', 'expenditure', 'owner',
            'selfemp', 'dependents', 'months', 'majorcards', 'active' ]

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Since there is no preprocessing, we don't need a pipeline (used anyway as best practice!)
my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))
cv_scores = cross_val_score(my_pipeline, X, y, 
                            cv=5,
                            scoring='accuracy')

print("Cross-validation accuracy: %f" % cv_scores.mean()) # 0.981052
```
It's very rare to have a model with 98% accuracy. It bears closer inspection of possible target leakage. Look at the summary of the data:
+ **card**: 1 if accepted, else 0
+ **reports**: number of derogatory reports
+ **age**: age n years plus twelfths of a year
+ **income**: yearly income divided by 10,000
+ **share**: ratio of monthly credit card expenditure to yearly income
+ **expenditure**: average monthly credit card expenditure
+ **owner**: 1 if owns home else 0
+ **selfemp**: 1 if self employed else 0
+ **dependents**: 1 + number of dependents
+ **months**: months living at current residence
+ **majorcards**: number of major credit cards held
+ **active**: number of active credit card accounts

What stands out? It's questionable whether a few of these variables are updated after `card=1`.  
For example, does expenditure mean on this card or on cards used before applying?
```angular2html
expenditures_cardholders = X.expenditure[y]
expenditures_noncardholders = X.expenditure[~y]

print('Fraction of those who did not receive a card and had no expenditures: %.2f' \
      %((expenditures_noncardholders == 0).mean())) # output: 1.00
print('Fraction of those who received a card and had no expenditures: %.2f' \
      %(( expenditures_cardholders == 0).mean())) # output: 0.02
```
Since everyone who did not receive a card had zero expenditure and 2% of card recipients had zero expenditures, we may assume this is a clear-cut case of target leakage.  

**Share** is dependent on **expenditure**, so it must also be excluded.  

**Active** and **majorcards** should be examined as well; alternatively, we may consider dropping them to ensure leakage is removed from the model without more information.
```angular2html
# Drop leaky predictors from dataset
potential_leaks = ['expenditure', 'share', 'active', 'majorcards']
X2 = X.drop(potential_leaks, axis=1)

# Evaluate the model with leaky predictors removed
cv_scores = cross_val_score(my_pipeline, X2, y, 
                            cv=5,
                            scoring='accuracy')

print("Cross-val accuracy: %f" % cv_scores.mean()) # Output: 0.830919
```
Even though this model only correctly predicts 83% of the time when used on new applications, the first model would likely do much worse. These types of mistakes can be expensive. Caution, consideration of "business sense", and data exploration are important mitigating methods.

---
