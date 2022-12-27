from src.classifiers.ada_boost import ada_boost
from test import test_1, test_2, test_3


def test_1_ada_boost():
    return test_1(ada_boost(), ['RowNumber', 'Surname'])


def test_2_ada_boost():
    return test_2(ada_boost())


def test_3_ada_boost():
    return test_3(ada_boost())
