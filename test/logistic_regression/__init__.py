from src.classifiers.logistic_regression import logistic_regression
from test import test_1, test_2


def test_1_logistic_regression():
    return test_1(logistic_regression())

def test_2_logistic_regression():
    return test_2(logistic_regression())