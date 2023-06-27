# Classifier imports
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

import os.path
import pylab as plt
import pandas as pd
import numpy as np

from joblib import dump, load

path_data = '../data/processed_data/data.csv'.replace('/',os.sep)
path_features = '../data/processed_data/features.csv'.replace('/',os.sep)
path_target = '../data/processed_data/target.csv'.replace('/',os.sep)

id_name = 'new_child'
trg_name = 'nomem_encr'

df = pd.read_csv(path_data)

# Create transformers.
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='infrequent_if_exist'))])

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

# Use ColumnTransformer to apply the transformations to the correct columns in the dataframe.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, selector(dtype_exclude=object)(df)),
        ('cat', cat_transformer, selector(dtype_include=object)(df))])

# Exporting features.
df.drop(columns = trg_name).to_csv(path_features, index = False)
# Exporting target.
df[[id_name,trg_name]].to_csv(path_target, index = False)
