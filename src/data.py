"""Data loading, preprocessing, and train/test splitting."""

import pathlib

import pandas as pd
from sklearn.model_selection import train_test_split

DROP_COLS = ["Unnamed: 0", "pid", "id", "fid", "mid"]
LABEL_COL = "crohn"
RANDOM_STATE = 42
TEST_SIZE = 0.30


def load_data(path: str | pathlib.Path = "data/crohn.csv") -> pd.DataFrame:
    """Load the raw Crohn's CSV and return a DataFrame."""
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop ID columns, binarise the label, and return (X, y)."""
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df[LABEL_COL] = (df[LABEL_COL] == 2).astype(int)
    y = df[LABEL_COL]
    X = df.drop(columns=[LABEL_COL])
    return X, y


def get_splits(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def load_and_split(
    path: str | pathlib.Path = "data/crohn.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Convenience: load → preprocess → split in one call."""
    df = load_data(path)
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = get_splits(X, y)
    print(f"Dataset loaded: {len(df)} rows, {X.shape[1]} features")
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"Label distribution (train): {y_train.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test
