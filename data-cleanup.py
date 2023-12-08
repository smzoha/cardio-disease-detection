import os

import pandas as pd
from mrmr import mrmr_classif

data = pd.read_csv('./data/cardio_train.csv', delimiter=';')
features = data[data.columns[1:-1]]
target = data[data.columns[-1]]

print('Number of empty rows:', data.isna().sum(), sep='\n')
print('=======================')
print('Number of duplicate rows:', data.duplicated().sum())
print('=======================')

print('Running mRMR for finding top 5 features')
selected_features = mrmr_classif(X=features, y=target, K=5)
print('Features selected by mRMR:', selected_features)

features = features[selected_features]

print('First 10 Rows of Features after selection:', features.head(20), sep='\n')
print('=======================')

features = (features - features.mean()) / features.std()
print('First 10 Rows of Features after standardization:', features.head(20), sep='\n')

if not os.path.exists('./data'):
    os.makedirs('./data')

print('Writing normalized dataset to ./data/cardio_train_updated.csv')
output_dataframe = pd.concat([features, target], axis=1)
output_dataframe.to_csv('./data/cardio_train_updated.csv', index=False)
print('Write successful!')
