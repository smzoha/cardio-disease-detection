import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB

from train_util import train_valid_test, split_train_test_data

data = pd.read_csv('../data/cardio_train_updated.csv')
x_train, x_test, y_train, y_test = split_train_test_data(data)

print('====== Multinomial Naive Bayes ======')
mnb_model = MultinomialNB()
train_valid_test(mnb_model, 'Multinomial Naive Bayes', x_train, y_train, x_test, y_test)

print('====== Gaussian Naive Bayes ======')
gnb_model = GaussianNB()
train_valid_test(gnb_model, 'Gaussian Naive Bayes', x_train, y_train, x_test, y_test)

print('====== Explaining Gaussian Naive Bayes ======')
explainer = LimeTabularExplainer(np.array(x_train), feature_names=x_train.columns, verbose=True)

print('First instance of test data:', x_test.iloc[0], sep='\n')
print('Corresponding true value:', y_test.iloc[0])
print('================')

explain_nb = explainer.explain_instance(x_test.iloc[0], gnb_model.predict_proba)
explain_nb.as_pyplot_figure()
plt.show()
