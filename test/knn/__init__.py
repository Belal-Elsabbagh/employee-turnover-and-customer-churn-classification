import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from src.tester import score_test
from test import preprocess_3, preprocess_2, preprocess_1, test_1
from sklearn.neighbors import KNeighborsClassifier


def test_1_knn():
    return test_1(KNeighborsClassifier(3), ['RowNumber', 'CustomerId', 'Surname'])
    df = pd.read_csv("data/1-customer-churn.csv")

    inputs = preprocess_1(df, ['RowNumber', 'CustomerId', 'Surname'])

    train, test = train_test_split(inputs, test_size=0.1)
    test_labels = test['Exited']
    test = test.drop('Exited', axis='columns')
    train_labels = train['Exited']
    train = train.drop('Exited', axis='columns')
    model = tree.DecisionTreeClassifier()
    model.fit(train, train_labels, check_input=True)
    prediction = model.predict(test)
    return score_test(prediction, test_labels)


def test_2_knn():
    df = pd.read_csv("data/2-hr-data.csv")

    inputs = preprocess_2(df, [])

    train, test = train_test_split(inputs, test_size=0.1)
    test_labels = test['left']
    test = test.drop('left', axis='columns')
    train_labels = train['left']
    train = train.drop('left', axis='columns')
    model = KNeighborsClassifier(3)
    model.fit(train, train_labels)
    prediction = model.predict(test)
    return score_test(prediction, test_labels)


def test_3_knn():
    df = pd.read_csv("data/3-telco-customer-churn.csv")

    inputs = preprocess_3(df, ['customerID'])

    train, test = train_test_split(inputs, test_size=0.1)
    test_labels = test['Churn']
    test = test.drop('Churn', axis='columns')
    train_labels = train['Churn']
    train = train.drop('Churn', axis='columns')
    model = KNeighborsClassifier(3)
    model.fit(train, train_labels)
    prediction = model.predict(test)
    return score_test(prediction, test_labels)