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

path_features = '../data/processed_data/features.csv'.replace('/',os.sep)
path_target = '.../data/processed_data/target.csv'.replace('/',os.sep)

df = pd.read_csv(path_features)
trg= pd.read_csv(path_target)

# Initialize the classifier.
model = GradientBoostingClassifier(random_state=42)

# Define the hyperparameter space to search
params = {
    'gradientboostingclassifier__learning_rate': [0.01, 0.1],
    'gradientboostingclassifier__n_estimators': [100],
    'gradientboostingclassifier__max_depth': [5],
    'gradientboostingclassifier__min_samples_split': [2],
    'gradientboostingclassifier__min_samples_leaf': [1]
}

# Initialize GridSearchCV.
grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1, scoring="f1", verbose=9)

# Fit the model.
grid_search.fit(df,  trg.values.ravel())

# Retrieve hyperparameters.
hyperparameters = grid_search.best_params_

print(hyperparameters)