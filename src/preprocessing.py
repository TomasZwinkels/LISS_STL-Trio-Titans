# Classifier imports
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer

import os.path
import pandas as pd

path_data = '../data/processed_data/data.csv'.replace('/',os.sep)
path_features = '../data/processed_data/features.csv'.replace('/',os.sep)
path_target = '../data/processed_data/target.csv'.replace('/',os.sep)

id_name = 'nomem_encr'
trg_name = 'new_child'

df = pd.read_csv(path_data)
df = df.set_index(id_name)
# Exporting target.
df[trg_name].to_csv(path_target)

df = df.drop(columns = trg_name)

cat_col = selector(dtype_include=object)(df)
num_col = selector(dtype_exclude=object)(df)

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
transformed_data = preprocessor.fit_transform(df).todense()

# Get the feature names of the transformed categorical columns
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_col)

# Create a list of all column names
new_column_names = num_col + list(cat_feature_names) 

# Convert the transformed data to a DataFrame with updated column names
df_transformed = pd.DataFrame(transformed_data, columns=new_column_names)

df_transformed = pd.DataFrame(index = df.index, 
                              columns = new_column_names,
                              data = transformed_data)
df_transformed.to_csv(path_features)




