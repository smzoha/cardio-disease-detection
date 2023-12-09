import os
import kaggle

if not os.path.exists('../data'):
    os.mkdir('../data/')

print('Authenticating...')
kaggle.api.authenticate()
print('Authentication successful!')

print('Downloading Dataset')
kaggle.api.dataset_download_files('sulianova/cardiovascular-disease-dataset', '../data', quiet=False, unzip=True)
print('Dataset downloaded!');
