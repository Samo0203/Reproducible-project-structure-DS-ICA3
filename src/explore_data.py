
import pandas as pd

from src.config import RAW_DATA_PATH


def explore_dataset():

    # Load the dataset
    df = pd.read_csv(RAW_DATA_PATH)

    print("DATASET INFORMATION")

    # Dataset dimensions
    print("\nDataset shape:")
    print(df.shape)

    # Column names
    print("\nColumn names:")
    print(df.columns.tolist())

    # Data types
    print("\nData types:")
    print(df.dtypes)

    # Missing values
    print("\nMissing values:")
    print(df.isnull().sum().sum())

    # Duplicate records
    print("\nDuplicate records:")
    print(df.duplicated().sum())

    # Statistical summary
    print("\nStatistical summary:")
    print(df.describe())

    # Target distribution
    print("\nTarget distribution:")
    print(df["target"].value_counts())

    print("\nDataset exploration completed!")


if __name__ == "__main__":
    explore_dataset()
