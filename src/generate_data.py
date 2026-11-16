
import pandas as pd

from sklearn.datasets import make_classification

from src.config import RAW_DATA_PATH, RANDOM_STATE


def generate_dataset():

    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.65, 0.35],
        flip_y=0.12,
        class_sep=0.6,
        random_state=RANDOM_STATE
    )

    # Create feature names
    columns = [f"feature_{i+1}" for i in range(X.shape[1])]

    # Convert data into a DataFrame
    df = pd.DataFrame(X, columns=columns)

    # Add target column
    df["target"] = y

    # Create directory if it does not exist
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save dataset
    df.to_csv(RAW_DATA_PATH, index=False)

    print("Dataset generated successfully!")
    print("Dataset shape:", df.shape)

    print("\nFirst five rows:")
    print(df.head())

    print("\nTarget distribution:")
    print(df["target"].value_counts())


if __name__ == "__main__":
    generate_dataset()
