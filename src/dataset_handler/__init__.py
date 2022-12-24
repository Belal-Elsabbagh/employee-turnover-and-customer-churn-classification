"""
This module handles the dataset operations
"""
import pandas as pd


def load_csv_dataset(csv_file_path: str) -> pd.DataFrame:
    """Loads a csv file as a pandas DataFrame

    Args:
        csv_file_path (str): The csv file path

    Returns:
        pd.DataFrame: the csv file's data as a DataFrame
    """
    return pd.read_csv(csv_file_path)
