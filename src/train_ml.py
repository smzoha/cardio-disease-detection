import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

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
svm_cv = CalibratedClassifierCV(svm)
train_valid_test(svm_cv, 'SVM', x_train, y_train, x_test, y_test)

print('====== Explaining SVM ======')
explainer = LimeTabularExplainer(np.array(x_train), feature_names=x_train.columns, verbose=True)

print('First instance of test data:', x_test.iloc[0], sep='\n')
print('Corresponding true value:', y_test.iloc[0])
print('================')

explain_nb = explainer.explain_instance(x_test.iloc[0], svm_cv.predict_proba)
explain_nb.as_pyplot_figure()
plt.show()
