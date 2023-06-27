import os
import pandas as pd

path_features = '../data/raw_data/LISS_example_input_data.csv'.replace('/',os.sep)
path_target = '../data/raw_data/LISS_example_groundtruth_data.csv'.replace('/',os.sep)
path_selection = '../data/processed_data/data.csv'.replace('/',os.sep)

df_orig = pd.read_csv(path_features,encoding="cp1252", low_memory=False)
trg = pd.read_csv(path_target)
trg = trg.dropna()
trg = trg.set_index('nomem_encr')

cols = ['geslacht',
        'nomem_encr',
        'burgstat2019',
        'leeftijd2019',
        'woonvorm2019',
        'oplmet2019',
        'aantalki2019']

df = df_orig[cols].copy()

df = df.merge(trg, on = 'nomem_encr')
df.to_csv(path_selection, index = False)