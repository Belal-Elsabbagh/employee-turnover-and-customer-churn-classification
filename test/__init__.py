import category_encoders as ce
import pandas as pd

from src.tester import base_test

count = -1


def preprocess_1(df, exclude_cols):
    df = df.loc[:, ~df.columns.isin(exclude_cols)]
    transformer = ce.OneHotEncoder(cols=['Gender', 'Geography'])
    return transformer.fit_transform(df)


def preprocess_2(df, exclude_cols):
    df = df.loc[:, ~df.columns.isin(exclude_cols)]
    df['salary'] = df['salary'].map({'low': 1, 'medium': 2, 'high': 3})
    transformer = ce.OneHotEncoder(cols=['department'])
    return transformer.fit_transform(df)


def preprocess_3(df: pd.DataFrame, exclude_cols):
    df = df.loc[:, ~df.columns.isin(exclude_cols)]
    df = df.replace({'Churn': {'Yes': 1, 'No': 0}})
    df['TotalCharges'] = df['TotalCharges'].replace(' ', '100')
    df['Contract'] = df['Contract'].map({'Month-to-month': 1, 'One year': 2, 'Two year': 3})
    df['TotalCharges'] = df['TotalCharges'].astype('float')
    transformer = ce.OneHotEncoder(
        cols=[
            'gender',
            'Partner',
            'Dependents',
            'PhoneService',
            'MultipleLines',
            'InternetService',
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies',
            'PaperlessBilling',
            'PaymentMethod',
        ]
    )
    return transformer.fit_transform(df)


def test_1(model, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
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
        exclude_cols = []
    file_path = r'data\2-hr-data.csv'
    target_col = 'left'
    index_col = None
    return base_test(
        file_path,
        model,
        index_col,
        target_col,
        test_size=0.15,
        exclude_cols=exclude_cols,
        preprocess=preprocess_2
    )


def test_3(model, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    file_path = r'data\3-telco-customer-churn.csv'
    target_col = 'Churn'
    index_col = 'customerID'
    return base_test(
        file_path,
        model,
        index_col,
        target_col,
        test_size=0.15,
        exclude_cols=exclude_cols,
        preprocess=preprocess_3
    )
