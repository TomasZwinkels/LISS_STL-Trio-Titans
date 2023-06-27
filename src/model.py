from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

import os.path
import pandas as pd

from joblib import dump

path_features = '../data/processed_data/features.csv'.replace('/',os.sep)
path_target = '../data/processed_data/target.csv'.replace('/',os.sep)
path_models = '../models/model.joblib'.replace('/',os.sep)

id_name = 'nomem_encr'
trg_name = 'new_child'

df = pd.read_csv(path_features).set_index(id_name)
trg = pd.read_csv(path_target).set_index(id_name)

#X_train, X_test, Y_train, Y_test = train_test_split(df, trg, random_state = 42, test_size = 0.2)

# Initialize the classifier.
model = GradientBoostingClassifier(random_state=42)

# Define the hyperparameter space to search
params = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100],
    'max_depth': [5],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

# Initialize GridSearchCV.
grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1, scoring="f1", verbose=9)

# Fit the model.
grid_search.fit(df,  trg.values.ravel())

# Retrieve hyperparameters.
hyperparameters = grid_search.best_params_

model = GradientBoostingClassifier(**hyperparameters)

model.fit(df,  trg.values.ravel())

# Dump model (don't change the name)
dump(model, path_models)

#print(hyperparameters)