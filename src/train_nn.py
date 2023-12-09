import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from train_util import train_valid_test, split_train_test_data

data = pd.read_csv('../data/cardio_train_updated.csv')
x_train, x_test, y_train, y_test = split_train_test_data(data)

print('====== Logistic Regression ======')
lr_model = LogisticRegression(max_iter=10000)
train_valid_test(lr_model, 'Logistic Regression', x_train, y_train, x_test, y_test)

print('====== Random Forest ======')
rf_model = RandomForestClassifier(n_estimators=25)
train_valid_test(rf_model, 'Random Forest', x_train, y_train, x_test, y_test)

print('====== SVM ======')
svm = LinearSVC(dual=False)
train_valid_test(svm, 'SVM', x_train, y_train, x_test, y_test)
