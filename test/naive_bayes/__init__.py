from sklearn.naive_bayes import GaussianNB
from test import test_1, test_2, test_3


def test_1_naive_bayes():
    return test_1(GaussianNB(), ['RowNumber', 'Surname'])


def test_2_naive_bayes():
    return test_2(GaussianNB())


def test_3_naive_bayes():
    return test_3(GaussianNB())
