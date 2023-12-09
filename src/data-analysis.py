import matplotlib.pyplot as plt
import pandas as pd

DATA_INFO = {
    'age': {'label': 'Age (in Days)', 'categorical': False},
    'gender': {'label': 'Gender', 'categorical': True},
    'height': {'label': 'Height (in cm)', 'categorical': False},
    'weight': {'label': 'Weight (kg)', 'categorical': False},
    'ap_hi': {'label': 'Systolic Blood Pressure', 'categorical': False},
    'ap_lo': {'label': 'Diastolic Blood Pressure', 'categorical': False},
    'cholesterol': {'label': 'Cholesterol', 'categorical': True},
    'gluc': {'label': 'Glucose', 'categorical': True},
    'smoke': {'label': 'Smoking', 'categorical': True},
    'alco': {'label': 'Alcohol Intake', 'categorical': True},
    'active': {'label': 'Physical Activity', 'categorical': True}
}


def plot_data(dataset, attr):
    info = DATA_INFO[attr]

    if info['categorical']:
        dataset[attr].value_counts().plot(kind='bar')
        plt.title(info['label'] + ' Distribution')

    else:
        dataset[attr].value_counts().plot(kind='hist')
        plt.title(info['label'] + ' Histogram')

    plt.xlabel(info['label'])
    plt.show()


data = pd.read_csv('../data/cardio_train.csv', delimiter=';')
features = data[data.columns[1:-1]]
target = data[data.columns[-1]]

print('Available features:', features.columns.values)
print('=======================\n')

for feature in features.columns:
    print(feature + ' Data Description:', features[feature].describe(), sep='\n')
    print('=======================\n')
    plot_data(features, feature)

print('Target (cardio) Data Description:', target.describe(), sep='\n')
target.value_counts().plot(kind='bar')
plt.xlabel('Cardio Vascular Disease')
plt.title('Cardio Vascular Disease Distribution')
plt.show()
