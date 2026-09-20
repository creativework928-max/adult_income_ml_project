"""
Dataset loading and validation utilities.
"""

from pathlib import Path

import pandas as pd

from ucimlrepo import fetch_ucirepo

from src.config import (
    UCI_DATASET_ID,
    TARGET_COLUMN,
    RAW_DATA_FILE,
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_adult_dataset():
    """
    Load the Adult dataset from the UCI Machine Learning Repository.

    Returns
    -------
    X : pandas.DataFrame
        Feature dataset.

    y : pandas.Series
        Target variable.

    metadata : object
        Dataset metadata.
    """

    adult = fetch_ucirepo(
        id=UCI_DATASET_ID
    )

    X = adult.data.features.copy()

    y = adult.data.targets.copy()

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    return X, y, adult.metadata


# ============================================================
# COMBINE DATA
# ============================================================

def combine_features_target(X, y):
    """
    Combine feature matrix and target into one DataFrame.
    """

    df = X.copy()

    df[TARGET_COLUMN] = y.values

    return df


# ============================================================
# SAVE RAW DATA
# ============================================================

def save_raw_dataset(df):
    """
    Save the downloaded raw dataset.
    """

    RAW_DATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        RAW_DATA_FILE,
        index=False
    )

    return RAW_DATA_FILE


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataset(df):
    """
    Perform basic structural validation.
    """

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "was not found."
        )

    expected_minimum_features = 14

    if len(df.columns) < expected_minimum_features:
        raise ValueError(
            "Dataset contains fewer features "
            "than expected."
        )

    return True


# ============================================================
# DATASET INFORMATION
# ============================================================

def get_dataset_summary(df):
    """
    Generate a high-level dataset summary.
    """

    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(
            df.isnull().sum().sum()
        ),
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
        "memory_mb": round(
            df.memory_usage(
                deep=True
            ).sum() / (1024 ** 2),
            2
        ),
    }

    return summary


# ============================================================
# COMPLETE LOADER
# ============================================================

def load_and_prepare_raw_data():
    """
    Download, combine, validate, and save the raw dataset.
    """

    X, y, metadata = load_adult_dataset()

    df = combine_features_target(
        X,
        y
    )

    validate_dataset(df)

    save_raw_dataset(df)

    return df, metadata


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    df, metadata = load_and_prepare_raw_data()

    print("=" * 70)
    print("ADULT DATASET")
    print("=" * 70)

    print(
        f"Rows: {df.shape[0]:,}"
    )

    print(
        f"Columns: {df.shape[1]}"
    )

    print(
        f"Missing cells: "
        f"{df.isnull().sum().sum():,}"
    )

    print(
        f"Duplicates: "
        f"{df.duplicated().sum():,}"
    )

    print(
        f"\nRaw dataset saved to:\n"
        f"{RAW_DATA_FILE}"
    )