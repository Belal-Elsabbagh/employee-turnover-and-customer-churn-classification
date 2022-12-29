from src.classifiers.svm import svm
from test import test_1, test_2, test_3


def test_1_svm():
    return test_1(svm(), ['RowNumber', 'Surname'])


def test_2_svm():
    return test_2(svm())


def test_3_svm():
    return test_3(svm())
