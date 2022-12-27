from src.tester import base_test


def preprocess_1(df):
    return df.replace({"Gender": {"Male": 1, "Female": 0}})


def test_1(model, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ['RowNumber', 'Surname', 'Geography']
    file_path = r'data\1-customer-churn.csv'
    target_col = 'Exited'
    index_col = 'CustomerId'
    return base_test(
        file_path,
        model,
        index_col,
        target_col,
        test_size=0.15,
        exclude_cols=exclude_cols,
        preprocess=preprocess_1
    )


def test_2(model, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ['department']
    file_path = r'data\2-hr-data.csv'
    target_col = 'left'
    index_col = None
    return base_test(
        file_path,
        model,
        index_col,
        target_col,
        test_size=0.15,
        exclude_cols=exclude_cols
    )
