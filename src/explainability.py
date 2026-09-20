"""
Production-oriented model explainability module.

Supports:
    - Random Forest feature_importances_
    - Logistic Regression coefficients
    - HistGradientBoosting permutation importance
    - Any sklearn Pipeline containing:
        * "preprocessor"
        * "model"

Outputs:
    reports/feature_importance.csv
    reports/feature_importance.json

For models without native feature importance, such as
HistGradientBoostingClassifier, permutation importance is used.

Important:
    Native feature importance is calculated on transformed
    features produced by the preprocessing pipeline.

    Permutation importance is calculated on the original
    input features by passing the complete sklearn Pipeline
    to permutation_importance().
"""

import json
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance


from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
    RANDOM_STATE,
)

from src.data_loader import (
    combine_features_target,
    load_adult_dataset,
)

from src.preprocessing import (
    clean_dataframe,
    split_dataset,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILENAME = "best_model.joblib"

IMPORTANCE_CSV_FILENAME = (
    "feature_importance.csv"
)

IMPORTANCE_JSON_FILENAME = (
    "feature_importance.json"
)

PERMUTATION_REPEATS = 10


# ============================================================
# DIRECTORIES
# ============================================================

def ensure_directories():
    """
    Ensure required output directories exist.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load the trained sklearn pipeline.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Trained model pipeline.
    """

    model_path = (
        MODELS_DIR /
        MODEL_FILENAME
    )

    if not model_path.exists():

        raise FileNotFoundError(
            "\n"
            "Trained model was not found.\n\n"
            f"Expected location:\n"
            f"{model_path}\n\n"
            "Run model training first:\n"
            "python -m src.model_training\n"
        )

    print(
        f"Loading model:\n"
        f"{model_path}"
    )

    try:

        model = joblib.load(
            model_path
        )

    except Exception as error:

        raise RuntimeError(
            "\n"
            "Unable to load the trained model.\n\n"
            "The model may have been created with an "
            "incompatible project version or an old "
            "custom transformer.\n\n"
            "FIX:\n"
            f"1. Delete:\n   {model_path}\n\n"
            "2. Retrain the model:\n"
            "   python -m src.model_training\n\n"
            "3. Run explainability again:\n"
            "   python -m src.explainability\n"
        ) from error

    return model


# ============================================================
# VALIDATE PIPELINE
# ============================================================

def get_pipeline_components(model):
    """
    Extract the preprocessing pipeline and estimator.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained sklearn pipeline.

    Returns
    -------
    tuple
        preprocessor, estimator
    """

    # --------------------------------------------------------
    # Check Pipeline
    # --------------------------------------------------------

    if not hasattr(
        model,
        "named_steps",
    ):

        raise TypeError(
            "The saved model is not an sklearn Pipeline."
        )

    # --------------------------------------------------------
    # Check preprocessor
    # --------------------------------------------------------

    if (
        "preprocessor"
        not in model.named_steps
    ):

        raise ValueError(
            "The saved pipeline does not contain "
            "a 'preprocessor' step."
        )

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if (
        "model"
        not in model.named_steps
    ):

        raise ValueError(
            "The saved pipeline does not contain "
            "a 'model' step."
        )

    preprocessor = (
        model.named_steps[
            "preprocessor"
        ]
    )

    estimator = (
        model.named_steps[
            "model"
        ]
    )

    return (
        preprocessor,
        estimator,
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():
    """
    Recreate the same deterministic test split used
    during model training.

    Returns
    -------
    tuple
        X_test, y_test
    """

    print(
        "\nLoading dataset..."
    )

    # --------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------

    X, y, _ = (
        load_adult_dataset()
    )

    # --------------------------------------------------------
    # Combine features and target
    # --------------------------------------------------------

    df = combine_features_target(
        X,
        y,
    )

    # --------------------------------------------------------
    # Clean data
    # --------------------------------------------------------

    df = clean_dataframe(
        df,
    )

    # --------------------------------------------------------
    # Recreate deterministic split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_dataset(
        df,
    )

    print(
        f"Test samples: "
        f"{len(X_test):,}"
    )

    return (
        X_test,
        y_test,
    )


# ============================================================
# FEATURE NAMES
# ============================================================

def get_feature_names(
    preprocessor,
    estimator=None,
):
    """
    Extract feature names generated by the preprocessing
    pipeline.

    These names correspond to transformed features such as
    one-hot encoded categorical variables.

    Parameters
    ----------
    preprocessor : sklearn ColumnTransformer
        Fitted preprocessing transformer.

    estimator : sklearn estimator, optional
        Model estimator.

    Returns
    -------
    list
        Feature names.
    """

    # --------------------------------------------------------
    # Preferred method
    # --------------------------------------------------------

    try:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        return list(
            feature_names
        )

    except Exception as error:

        print(
            "\nWarning: unable to obtain transformed "
            "feature names."
        )

        print(
            f"Reason: {error}"
        )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    n_features = 0

    if estimator is not None:

        if hasattr(
            estimator,
            "feature_importances_",
        ):

            n_features = len(
                estimator
                .feature_importances_
            )

        elif hasattr(
            estimator,
            "coef_",
        ):

            coefficients = np.asarray(
                estimator.coef_
            )

            n_features = (
                coefficients.shape[-1]
            )

    return [
        f"feature_{index}"
        for index in range(
            n_features
        )
    ]


# ============================================================
# NATIVE FEATURE IMPORTANCE
# ============================================================

def extract_native_importance(
    estimator,
    feature_names,
):
    """
    Extract native feature importance when available.

    Supported:
        - Random Forest
        - Tree-based models exposing
          feature_importances_
        - Logistic Regression
        - Linear models exposing coef_

    Parameters
    ----------
    estimator : sklearn estimator
        Trained estimator.

    feature_names : list
        Transformed feature names.

    Returns
    -------
    pandas.DataFrame or None
        Feature importance results, or None when the model
        does not provide native importance.
    """

    # ========================================================
    # TREE-BASED FEATURE IMPORTANCE
    # ========================================================

    if hasattr(
        estimator,
        "feature_importances_",
    ):

        importance = np.asarray(
            estimator.feature_importances_,
            dtype=float,
        )

        if len(feature_names) != len(
            importance
        ):

            raise ValueError(
                "Number of feature names does not match "
                "number of feature importances.\n"
                f"Feature names: "
                f"{len(feature_names)}\n"
                f"Importances: "
                f"{len(importance)}"
            )

        result = pd.DataFrame(
            {
                "feature": feature_names,

                "importance": importance,

                "importance_std": np.nan,

                "method":
                    "native_feature_importance",
            }
        )

        return result

    # ========================================================
    # LINEAR MODEL COEFFICIENTS
    # ========================================================

    if hasattr(
        estimator,
        "coef_",
    ):

        coefficients = np.asarray(
            estimator.coef_,
            dtype=float,
        )

        # ----------------------------------------------------
        # Binary classification
        # ----------------------------------------------------

        if coefficients.ndim == 2:

            if coefficients.shape[0] == 1:

                coefficients = (
                    coefficients[0]
                )

            else:

                # Multiclass case:
                # use the mean absolute coefficient
                coefficients = np.mean(
                    np.abs(
                        coefficients
                    ),
                    axis=0,
                )

        # ----------------------------------------------------
        # Absolute coefficient magnitude
        # ----------------------------------------------------

        importance = np.abs(
            coefficients
        )

        if len(feature_names) != len(
            importance
        ):

            raise ValueError(
                "Number of feature names does not match "
                "number of model coefficients.\n"
                f"Feature names: "
                f"{len(feature_names)}\n"
                f"Coefficients: "
                f"{len(importance)}"
            )

        result = pd.DataFrame(
            {
                "feature": feature_names,

                "importance": importance,

                "importance_std": np.nan,

                "method":
                    "absolute_model_coefficient",
            }
        )

        return result

    # ========================================================
    # NO NATIVE IMPORTANCE
    # ========================================================

    return None


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

def extract_permutation_importance(
    model,
    X_test,
    y_test,
):
    """
    Calculate permutation importance directly on the
    complete sklearn pipeline.

    This is especially useful for models such as:
        HistGradientBoostingClassifier

    Because the complete pipeline is passed to
    permutation_importance(), the original unprocessed
    DataFrame can be used.

    The resulting importance is calculated for ORIGINAL
    INPUT FEATURES, rather than individual one-hot encoded
    columns.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Complete trained pipeline.

    X_test : pandas.DataFrame
        Original test features.

    y_test : pandas.Series
        Test target.

    Returns
    -------
    pandas.DataFrame
        Permutation importance results.
    """

    print(
        "\nCalculating permutation importance..."
    )

    print(
        f"Repeats: "
        f"{PERMUTATION_REPEATS}"
    )

    print(
        "This may take a little while..."
    )

    # --------------------------------------------------------
    # Calculate permutation importance
    # --------------------------------------------------------

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # --------------------------------------------------------
    # Original input feature names
    # --------------------------------------------------------

    feature_names = list(
        X_test.columns
    )

    importances_mean = (
        result.importances_mean
    )

    importances_std = (
        result.importances_std
    )

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if len(feature_names) != len(
        importances_mean
    ):

        raise ValueError(
            "Permutation importance output does not match "
            "the number of input features.\n"
            f"Input features: "
            f"{len(feature_names)}\n"
            f"Importances: "
            f"{len(importances_mean)}"
        )

    # --------------------------------------------------------
    # Create result DataFrame
    # --------------------------------------------------------

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,

            "importance":
                importances_mean,

            "importance_std":
                importances_std,

            "method":
                "permutation_importance",
        }
    )

    return importance_df


# ============================================================
# SORT IMPORTANCE
# ============================================================

def sort_importance(
    result,
):
    """
    Sort feature importance from highest to lowest.

    Parameters
    ----------
    result : pandas.DataFrame
        Feature importance results.

    Returns
    -------
    pandas.DataFrame
        Sorted feature importance.
    """

    result = result.copy()

    result = (
        result
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True,
        )
    )

    return result


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    result,
):
    """
    Save feature importance to CSV.

    Returns
    -------
    pathlib.Path
        Output path.
    """

    output_path = (
        REPORTS_DIR /
        IMPORTANCE_CSV_FILENAME
    )

    result.to_csv(
        output_path,
        index=False,
    )

    return output_path


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    result,
):
    """
    Save feature importance to JSON.

    Returns
    -------
    pathlib.Path
        Output path.
    """

    output_path = (
        REPORTS_DIR /
        IMPORTANCE_JSON_FILENAME
    )

    records = []

    for record in result.to_dict(
        orient="records"
    ):

        clean_record = {}

        for key, value in record.items():

            if isinstance(
                value,
                np.generic,
            ):

                value = value.item()

            # ------------------------------------------------
            # Convert NaN to None for valid JSON
            # ------------------------------------------------

            if isinstance(
                value,
                float,
            ) and np.isnan(value):

                value = None

            clean_record[key] = value

        records.append(
            clean_record
        )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            records,
            file,
            indent=4,
        )

    return output_path


# ============================================================
# MAIN EXPLAINABILITY WORKFLOW
# ============================================================

def extract_feature_importance():
    """
    Main model explainability workflow.

    Strategy:

        1. Load the trained pipeline.
        2. Validate pipeline structure.
        3. Check for native feature importance.
        4. Use native importance when available.
        5. Otherwise use permutation importance.
        6. Sort the results.
        7. Save CSV and JSON reports.

    Returns
    -------
    pandas.DataFrame
        Feature importance results.
    """

    print("=" * 70)
    print("MODEL EXPLAINABILITY")
    print("=" * 70)

    # --------------------------------------------------------
    # Ensure directories
    # --------------------------------------------------------

    ensure_directories()

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Extract pipeline components
    # --------------------------------------------------------

    (
        preprocessor,
        estimator,
    ) = get_pipeline_components(
        model
    )

    estimator_name = (
        estimator.__class__.__name__
    )

    print(
        f"\nEstimator: "
        f"{estimator_name}"
    )

    # ========================================================
    # TRY NATIVE FEATURE IMPORTANCE
    # ========================================================

    feature_names = get_feature_names(
        preprocessor,
        estimator,
    )

    result = extract_native_importance(
        estimator,
        feature_names,
    )

    if result is not None:

        print(
            "\nImportance method:"
            " native model importance"
        )

    # ========================================================
    # FALLBACK TO PERMUTATION IMPORTANCE
    # ========================================================

    else:

        print(
            "\nEstimator does not expose "
            "feature_importances_ or coef_."
        )

        print(
            "Using permutation importance."
        )

        (
            X_test,
            y_test,
        ) = load_test_data()

        result = (
            extract_permutation_importance(
                model,
                X_test,
                y_test,
            )
        )

    # --------------------------------------------------------
    # Sort results
    # --------------------------------------------------------

    result = sort_importance(
        result
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = save_csv(
        result
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    json_path = save_json(
        result
    )

    # ========================================================
    # DISPLAY TOP FEATURES
    # ========================================================

    print()
    print("=" * 70)
    print("TOP FEATURES")
    print("=" * 70)

    print(
        result.head(20).to_string(
            index=False,
        )
    )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    print()
    print("=" * 70)
    print("FILES GENERATED")
    print("=" * 70)

    print(
        f"CSV:\n"
        f"{csv_path}"
    )

    print(
        f"JSON:\n"
        f"{json_path}"
    )

    print()
    print("=" * 70)
    print("EXPLAINABILITY COMPLETE")
    print("=" * 70)

    return result


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    extract_feature_importance()
