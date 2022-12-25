"""
This module handles the dataset operations
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def default_preprocess(df):
    return df


def load_csv_dataset(
        csv_file_path: str,
        index_col: str,
        target_col: str,
        exclude_cols: list | None = None,
        test_size: float = 0.2,
        preprocess: callable = default_preprocess
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the dataset from the csv file, splits target from features and splits them into train and test sets

    Args:
        preprocess (callable):
        test_size (float): The proportion of data to test. Defaults to 0.2
        csv_file_path (str): The path of the csv file
        index_col (str): The index column name
        target_col (str): The target column name
        exclude_cols (list, optional): Names of columns to exclude. Defaults to an empty array.

    Returns:
        tuple[ pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: _description_
    """
    if exclude_cols is None:
        exclude_cols = []
    df: pd.DataFrame = pd.read_csv(csv_file_path, index_col=index_col)
    df = preprocess(df)
    return train_test_split(
        df.loc[:, ~df.columns.isin([target_col, index_col] + exclude_cols)],
        df.loc[:, df.columns.isin([target_col])],
        test_size=test_size,
        random_state=42
    )
