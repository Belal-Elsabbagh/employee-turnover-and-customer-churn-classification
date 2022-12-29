"""
The main module. Execution starts here
"""
import json

from test import logistic_regression, ada_boost, svm, naive_bayes

if __name__ == '__main__':
    results = {
        'LogisticRegression': {
            '1-customer-churn.csv': logistic_regression.test_1_logistic_regression(),
            '2-hr-data.csv': logistic_regression.test_2_logistic_regression(),
            '3-telco-customer-churn.csv': logistic_regression.test_3_logistic_regression()
        },
        'AdaBoostClassifier': {
            '1-customer-churn.csv': ada_boost.test_1_ada_boost(),
            '2-hr-data.csv': ada_boost.test_2_ada_boost(),
            '3-telco-customer-churn.csv': ada_boost.test_3_ada_boost()
        },
        'GaussianNB': {
            '1-customer-churn.csv': naive_bayes.test_1_naive_bayes(),
            '2-hr-data.csv': naive_bayes.test_2_naive_bayes(),
            '3-telco-customer-churn.csv': naive_bayes.test_3_naive_bayes(),
        },
        'SVC': {
            '1-customer-churn.csv': svm.test_1_svm(),
            '2-hr-data.csv': svm.test_2_svm(),
            '3-telco-customer-churn.csv': svm.test_3_svm(),
        }
    }
    with open('out\\test-results\\all-results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
