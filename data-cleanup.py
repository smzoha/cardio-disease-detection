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
