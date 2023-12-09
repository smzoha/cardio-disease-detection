from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split


def train_valid_test(model, title, train_x, train_y, test_x, test_y):
    print('Training', title)
    model.fit(train_x, train_y)

    score = cross_val_score(model, train_x, train_y, scoring='accuracy', cv=100)
    print('Mean Accuracy: %.2f' % score.mean())

    plt.plot(score)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Cross-validation score for accuracy of ' + title)
    plt.show()

    y_pred = model.predict(test_x)
    report = classification_report(y_true=test_y, y_pred=y_pred)

    print(report)
    print('Mean-Squared Error for', title, ': %.2f', mean_squared_error(test_y, y_pred))


def split_train_test_data(data):
    x = data[data.columns[:-1]]
    y = data[data.columns[-1]]

    print('Splitting data for training and testing (70-30 split)')
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

    return x_train, x_test, y_train, y_test
