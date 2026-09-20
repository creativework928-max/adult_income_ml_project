import pandas as pd

from src.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)


EXPECTED_FEATURES = (
    NUMERIC_FEATURES +
    CATEGORICAL_FEATURES
)


def validate_schema(df: pd.DataFrame):
    """
    Validate required dataset columns.
    """

    required = EXPECTED_FEATURES + [TARGET_COLUMN]

    missing_columns = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    return True


def validate_numeric_columns(df):

    errors = []

    for column in NUMERIC_FEATURES:

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            errors.append(
                f"{column} is not numeric"
            )

    if errors:
        raise ValueError(
            "\n".join(errors)
        )

    return True


def validate_target(df):

    if df[TARGET_COLUMN].isna().all():
        raise ValueError(
            "Target column contains no valid values."
        )

    return True


def validate_dataset(df):

    validate_schema(df)

    validate_numeric_columns(df)

    validate_target(df)

    return {
        "valid": True,
        "rows": len(df),
        "columns": len(df.columns),
    }