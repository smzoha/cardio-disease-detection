import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import MinMaxScaler


def train_valid_test(model, title, train_x, train_y, test_x, test_y):
    print('Training', title)
    model.fit(train_x, train_y)

    score = cross_val_score(model, train_x, train_y, scoring='accuracy', cv=100)
    print('Mean Accuracy: %.2f' % score.mean())

    plt.plot(score)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Cross-validation score for accuracy of', title)
    plt.show()

    y_pred = model.predict(test_x)
    report = classification_report(y_true=test_y, y_pred=y_pred)

    print(report)
    print('Mean-Squared Error for', title, ': %.2f', mean_squared_error(test_y, y_pred))


data = pd.read_csv('./data/cardio_train_updated.csv')

X = data[data.columns[:-1]]
y = data[data.columns[-1]]

print('Splitting data for training and testing (70-30 split)')
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

print('Normalizing data for training using MinxMaxScaler')
scaler = MinMaxScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print('====== Multinomial Naive Bayes ======')
mnb_model = MultinomialNB()
train_valid_test(mnb_model, 'Multinomial Naive Bayes', x_train, y_train, x_test, y_test)

print('====== Gaussian Naive Bayes ======')
gnb_model = GaussianNB()
train_valid_test(gnb_model, 'Gaussian Naive Bayes', x_train, y_train, x_test, y_test)

print('====== Logistic Regression ======')
lr_model = LogisticRegression(max_iter=10000)
train_valid_test(lr_model, 'Logistic Regression', x_train, y_train, x_test, y_test)

print('====== Random Forest ======')
rf_model = RandomForestClassifier(n_estimators=25)
train_valid_test(rf_model, 'Random Forest', x_train, y_train, x_test, y_test)
