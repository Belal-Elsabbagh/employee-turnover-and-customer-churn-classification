"""
The main module. Execution starts here
"""
from src.dataset_handler import load_csv_dataset


if __name__ == '__main__':
    print('Started Execution')
    df = load_csv_dataset(r'data\customer-churn.csv')
    print(df)
