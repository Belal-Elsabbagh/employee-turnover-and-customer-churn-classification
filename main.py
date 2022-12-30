"""
The main module. Execution starts here
"""
import json

from test import logistic_regression, ada_boost, knn

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
         'KNeighborsClassifier': {
            '1-customer-churn.csv': knn.test_1_knn(),
            '2-hr-data.csv': knn.test_2_knn(),
            '3-telco-customer-churn.csv': knn.test_3_knn()
        }
    }
    with open('out\\test-results\\all-results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
