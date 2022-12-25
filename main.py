"""
The main module. Execution starts here
"""
from test import logistic_regression, svm


if __name__ == '__main__':
    print('Logistic Regression')
    logistic_regression.test_1_logistic_regression()
    print('---------------------------')
    print('SVM')
    svm.test_1_svm()
    print('---------------------------')
