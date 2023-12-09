import pandas as pd
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
