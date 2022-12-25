from src.tester.base_test import base_test

def preprocess_1(df):
    return df.replace({"Gender": {"Male": 1, "Female": 0}})

def test_1(model):
    customer_churn_path = r'data\1-customer-churn.csv'
    target_col = 'Exited'
    index_col = 'CustomerId'
    base_test(
        customer_churn_path,
        model,
        index_col,
        target_col,
        test_size=0.15,
        exclude_cols=['RowNumber', 'Surname', 'Geography'],
        preprocess=preprocess_1
    )

def test_2(model):
    customer_churn_path = r'data\2-hr-data.csv'
    target_col = 'left'
    index_col = None
    base_test(
        customer_churn_path,
        model,
        index_col,
        target_col,
        test_size=0.15,
        exclude_cols=['department']
    )