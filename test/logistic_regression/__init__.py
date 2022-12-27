from src.classifiers.logistic_regression import logistic_regression
from test import test_1, test_2, test_3


def test_1_logistic_regression():
    return test_1(logistic_regression(), ['RowNumber', 'Surname'])

def test_2_logistic_regression():
    return test_2(logistic_regression())

def test_3_logistic_regression():
    return test_3(logistic_regression())