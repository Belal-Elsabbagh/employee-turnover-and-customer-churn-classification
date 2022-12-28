from src.classifiers.naive_bayes import naive_bayes
from test import test_1, test_2, test_3


def test_1_naive_bayes():
    return test_1(naive_bayes(), ['RowNumber', 'Surname'])


def test_2_naive_bayes():
    return test_2(naive_bayes())


def test_3_naive_bayes():
    return test_3(naive_bayes())