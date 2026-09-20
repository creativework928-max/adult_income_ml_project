"""
Production-oriented preprocessing pipeline.

This module:
    1. Cleans the raw dataset.
    2. Encodes the target variable.
    3. Splits data into training and testing sets.
    4. Builds a leakage-safe preprocessing pipeline.
    5. Handles missing numerical and categorical values.
    6. Clips numerical outliers using training quantiles.
    7. Standardizes numerical features.
    8. One-hot encodes categorical features.
    9. Saves the fitted preprocessing pipeline.
"""

import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    OUTLIER_LOWER_QUANTILE,
    OUTLIER_UPPER_QUANTILE,
    PIPELINE_FILE,
    PREPROCESSOR_PATH,
)


# ============================================================
# DATA CLEANING
# ============================================================

def clean_dataframe(df):
    """
    Perform deterministic cleaning.

    Operations:
        - Replace '?' with NaN.
        - Strip whitespace from object columns.
        - Normalize target labels.
        - Ensure missing values are represented by np.nan.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw input dataframe.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Replace common missing-value markers
    # --------------------------------------------------------

    df = df.replace(
        [
            "?",
            " ?",
            "? ",
        ],
        np.nan,
    )

    # --------------------------------------------------------
    # Strip whitespace from categorical/object columns
    # --------------------------------------------------------

    object_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in object_columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

        # Convert string representations of missing values
        # back to actual numpy NaN values.
        df[column] = df[column].replace(
            {
                "nan": np.nan,
                "None": np.nan,
                "<NA>": np.nan,
                "?": np.nan,
                "": np.nan,
            }
        )

    # --------------------------------------------------------
    # Normalize target
    # --------------------------------------------------------

    if TARGET_COLUMN in df.columns:

        df[TARGET_COLUMN] = (
            df[TARGET_COLUMN]
            .astype(str)
            .str.strip()
            .str.replace(
                ".",
                "",
                regex=False,
            )
        )

        df[TARGET_COLUMN] = df[TARGET_COLUMN].replace(
            {
                "nan": np.nan,
                "None": np.nan,
                "<NA>": np.nan,
                "": np.nan,
            }
        )

    # --------------------------------------------------------
    # Final missing-value normalization
    # --------------------------------------------------------

    df = df.replace(
        {
            pd.NA: np.nan,
        }
    )

    return df


# ============================================================
# TARGET ENCODING
# ============================================================

def encode_target(y):
    """
    Convert income labels into binary values.

    Mapping:
        <=50K -> 0
        >50K  -> 1

    Parameters
    ----------
    y : pandas.Series
        Target labels.

    Returns
    -------
    pandas.Series
        Integer-encoded target.
    """

    y = (
        y.astype(str)
        .str.strip()
        .str.replace(
            ".",
            "",
            regex=False,
        )
    )

    mapping = {
        "<=50K": 0,
        ">50K": 1,
    }

    encoded = y.map(mapping)

    if encoded.isna().any():

        unknown = (
            y[encoded.isna()]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unknown target labels: {unknown}"
        )

    return encoded.astype(int)


# ============================================================
# ROBUST QUANTILE OUTLIER CLIPPER
# ============================================================

class RobustQuantileClipper(
    BaseEstimator,
    TransformerMixin,
):
    """
    Clip numerical values between lower and upper
    training quantiles.

    Quantiles are learned ONLY during fit().
    """

    def __init__(
        self,
        lower_quantile=0.01,
        upper_quantile=0.99,
    ):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(
        self,
        X,
        y=None,
    ):
        """
        Learn clipping bounds from training data only.
        """

        # ----------------------------------------------------
        # Preserve feature names
        # ----------------------------------------------------

        if isinstance(X, pd.DataFrame):

            self.feature_names_in_ = np.asarray(
                X.columns,
                dtype=object,
            )

            data = X.copy()

        else:

            data = pd.DataFrame(X)

            self.feature_names_in_ = np.asarray(
                [
                    f"x{i}"
                    for i in range(data.shape[1])
                ],
                dtype=object,
            )

        # ----------------------------------------------------
        # Convert to numeric values
        # ----------------------------------------------------

        data = data.apply(
            pd.to_numeric,
            errors="coerce",
        )

        # ----------------------------------------------------
        # Calculate quantiles ONLY on training data
        # ----------------------------------------------------

        self.lower_bounds_ = (
            data.quantile(
                self.lower_quantile,
            )
            .to_numpy(dtype=float)
        )

        self.upper_bounds_ = (
            data.quantile(
                self.upper_quantile,
            )
            .to_numpy(dtype=float)
        )

        return self

    def transform(self, X):
        """
        Apply learned clipping bounds.
        """

        if isinstance(X, pd.DataFrame):

            data = X.copy()

        else:

            data = pd.DataFrame(
                X,
                columns=self.feature_names_in_,
            )

        # ----------------------------------------------------
        # Convert to numeric
        # ----------------------------------------------------

        data = data.apply(
            pd.to_numeric,
            errors="coerce",
        )

        values = data.to_numpy(
            dtype=float,
        )

        # ----------------------------------------------------
        # Apply learned clipping bounds
        # ----------------------------------------------------

        clipped = np.clip(
            values,
            self.lower_bounds_,
            self.upper_bounds_,
        )

        return clipped

    def get_feature_names_out(
        self,
        input_features=None,
    ):
        """
        Return unchanged feature names.

        Required by scikit-learn when this transformer
        is used inside a Pipeline/ColumnTransformer.
        """

        if input_features is not None:

            return np.asarray(
                input_features,
                dtype=object,
            )

        return np.asarray(
            self.feature_names_in_,
            dtype=object,
        )


# ============================================================
# BUILD PREPROCESSING PIPELINE
# ============================================================

def build_preprocessing_pipeline():
    """
    Build the complete preprocessing pipeline.

    Numerical:
        1. Median imputation
        2. Quantile outlier clipping
        3. Standard scaling

    Categorical:
        1. Most-frequent imputation
        2. One-hot encoding

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Complete preprocessing pipeline.
    """

    # --------------------------------------------------------
    # Numerical preprocessing
    # --------------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "median_imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "outlier_clipper",
                RobustQuantileClipper(
                    lower_quantile=(
                        OUTLIER_LOWER_QUANTILE
                    ),
                    upper_quantile=(
                        OUTLIER_UPPER_QUANTILE
                    ),
                ),
            ),
            (
                "standard_scaler",
                StandardScaler(),
            ),
        ]
    )

    # --------------------------------------------------------
    # Categorical preprocessing
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "most_frequent_imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "one_hot_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # Combined preprocessing
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_dataset(df):
    """
    Split dataset using stratification.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned dataframe.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = encode_target(
        df[TARGET_COLUMN]
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# FIT + TRANSFORM
# ============================================================

def preprocess_data(df):
    """
    Split and preprocess data without leakage.

    The preprocessing pipeline is fitted ONLY on
    the training dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataframe.

    Returns
    -------
    tuple
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_processed,
        X_test_processed,
        preprocessor
    """

    # --------------------------------------------------------
    # Clean data
    # --------------------------------------------------------

    df = clean_dataframe(df)

    # --------------------------------------------------------
    # Split data
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_dataset(df)

    # --------------------------------------------------------
    # Build preprocessing pipeline
    # --------------------------------------------------------

    preprocessor = build_preprocessing_pipeline()

    # --------------------------------------------------------
    # Fit ONLY on training data
    # --------------------------------------------------------

    X_train_processed = (
        preprocessor.fit_transform(
            X_train,
            y_train,
        )
    )

    # --------------------------------------------------------
    # Transform test data
    # --------------------------------------------------------

    X_test_processed = (
        preprocessor.transform(
            X_test,
        )
    )

    # --------------------------------------------------------
    # Save fitted preprocessing pipeline
    # --------------------------------------------------------

    joblib.dump(
        preprocessor,
        PIPELINE_FILE,
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_processed,
        X_test_processed,
        preprocessor,
    )


# ============================================================
# FEATURE NAMES
# ============================================================

def get_processed_feature_names(
    preprocessor,
):
    """
    Return feature names after preprocessing.

    Parameters
    ----------
    preprocessor : sklearn ColumnTransformer
        Fitted preprocessing pipeline.

    Returns
    -------
    numpy.ndarray
        Processed feature names.
    """

    return preprocessor.get_feature_names_out()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    from src.data_loader import (
        load_and_prepare_raw_data,
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df, _ = load_and_prepare_raw_data()

    # --------------------------------------------------------
    # Preprocess dataset
    # --------------------------------------------------------

    results = preprocess_data(df)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_processed,
        X_test_processed,
        preprocessor,
    ) = results

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    feature_names = get_processed_feature_names(
        preprocessor
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Training samples: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test):,}"
    )

    print(
        f"Processed training shape: "
        f"{X_train_processed.shape}"
    )

    print(
        f"Processed testing shape: "
        f"{X_test_processed.shape}"
    )

    print(
        f"Processed features: "
        f"{len(feature_names):,}"
    )

    print(
        f"Pipeline saved to:\n"
        f"{PIPELINE_FILE}"
    )

    print(
        f"Preprocessor saved to:\n"
        f"{PREPROCESSOR_PATH}"
    )
