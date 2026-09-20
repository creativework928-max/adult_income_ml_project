"""
Production-oriented model training module.

Trains multiple classification models using leakage-safe
preprocessing pipelines and saves the best model.

Supported models:
    - Logistic Regression
    - Random Forest
    - HistGradientBoosting

Important:
    HistGradientBoostingClassifier requires dense input,
    while LogisticRegression and RandomForestClassifier
    can work with sparse one-hot encoded data.

Therefore, HistGradientBoosting uses a separate preprocessing
pipeline configured with a dense OneHotEncoder.
"""

import json
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from src.config import (
    CATEGORICAL_FEATURES,
    MODELS_DIR,
    NUMERIC_FEATURES,
    OUTLIER_LOWER_QUANTILE,
    OUTLIER_UPPER_QUANTILE,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
)

from src.data_loader import (
    combine_features_target,
    load_adult_dataset,
)

from src.preprocessing import (
    RobustQuantileClipper,
    clean_dataframe,
    split_dataset,
)


# ============================================================
# DIRECTORIES
# ============================================================

def ensure_directories():
    """
    Create model and report directories if they do not exist.
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
# SPARSE PREPROCESSING
# ============================================================

def build_sparse_preprocessing_pipeline():
    """
    Build preprocessing for models that support sparse matrices.

    Used by:
        - Logistic Regression
        - Random Forest

    Returns
    -------
    ColumnTransformer
        Sparse-compatible preprocessing pipeline.
    """

    # --------------------------------------------------------
    # Numerical pipeline
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
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
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    # --------------------------------------------------------
    # Categorical pipeline
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
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
                "numeric",
                numeric_pipeline,
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
# DENSE PREPROCESSING
# ============================================================

def build_dense_preprocessing_pipeline():
    """
    Build preprocessing for HistGradientBoostingClassifier.

    HistGradientBoostingClassifier requires dense input.

    Therefore, the OneHotEncoder explicitly returns a dense
    NumPy array.

    Returns
    -------
    ColumnTransformer
        Dense preprocessing pipeline.
    """

    # --------------------------------------------------------
    # Numerical pipeline
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
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
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    # --------------------------------------------------------
    # Categorical pipeline
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
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
                "numeric",
                numeric_pipeline,
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
# MODELS
# ============================================================

def get_models():
    """
    Return the model configurations.

    Each model receives its own preprocessing pipeline.

    Returns
    -------
    dict
        Model configurations.
    """

    return {
        "Logistic Regression": {
            "preprocessor": (
                build_sparse_preprocessing_pipeline()
            ),
            "model": LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        },

        "Random Forest": {
            "preprocessor": (
                build_sparse_preprocessing_pipeline()
            ),
            "model": RandomForestClassifier(
                n_estimators=300,
                max_depth=18,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },

        "HistGradientBoosting": {
            "preprocessor": (
                build_dense_preprocessing_pipeline()
            ),
            "model": HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.08,
                max_leaf_nodes=31,
                random_state=RANDOM_STATE,
            ),
        },
    }


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_single_model(
    name,
    configuration,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Train and evaluate one model.

    Parameters
    ----------
    name : str
        Model name.

    configuration : dict
        Model and preprocessing configuration.

    X_train : pandas.DataFrame
        Training features.

    X_test : pandas.DataFrame
        Testing features.

    y_train : pandas.Series
        Training target.

    y_test : pandas.Series
        Testing target.

    Returns
    -------
    tuple
        Fitted pipeline and metrics dictionary.
    """

    print()
    print("-" * 70)
    print(
        f"Training: {name}"
    )
    print("-" * 70)

    # --------------------------------------------------------
    # Build pipeline
    # --------------------------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                configuration["preprocessor"],
            ),
            (
                "model",
                configuration["model"],
            ),
        ]
    )

    # --------------------------------------------------------
    # Train ONLY on training data
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Generate probabilities
    # --------------------------------------------------------

    probabilities = (
        pipeline
        .predict_proba(
            X_test,
        )[:, 1]
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    metrics = {
        "model": name,

        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print(
        f"Accuracy : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1       : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{metrics['roc_auc']:.4f}"
    )

    return (
        pipeline,
        metrics,
    )


# ============================================================
# TRAIN ALL MODELS
# ============================================================

def train_models():
    """
    Train all configured models and save the best one.

    The preprocessing pipeline for every model is fitted
    exclusively on X_train.

    Returns
    -------
    pandas.DataFrame
        Model comparison results.
    """

    print("=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    ensure_directories()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        "\nLoading dataset..."
    )

    X, y, metadata = load_adult_dataset()

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
    # Split data
    # --------------------------------------------------------

    print(
        "Creating deterministic train/test split..."
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_dataset(
        df,
    )

    print(
        f"Training samples: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test):,}"
    )

    # --------------------------------------------------------
    # Get model configurations
    # --------------------------------------------------------

    models = get_models()

    results = []

    best_pipeline = None
    best_score = -np.inf
    best_name = None

    # --------------------------------------------------------
    # Train every model
    # --------------------------------------------------------

    for name, configuration in models.items():

        (
            pipeline,
            metrics,
        ) = evaluate_single_model(
            name=name,
            configuration=configuration,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        results.append(
            metrics
        )

        # ----------------------------------------------------
        # Select best model using ROC-AUC
        # ----------------------------------------------------

        if metrics["roc_auc"] > best_score:

            best_score = metrics["roc_auc"]

            best_pipeline = pipeline

            best_name = name

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if best_pipeline is None:

        raise RuntimeError(
            "No model was successfully trained."
        )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "roc_auc",
            ascending=False,
        )
        .reset_index(
            drop=True,
        )
    )

    # --------------------------------------------------------
    # Save comparison report
    # --------------------------------------------------------

    comparison_path = (
        REPORTS_DIR /
        "model_comparison.csv"
    )

    results_df.to_csv(
        comparison_path,
        index=False,
    )

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    model_path = (
        MODELS_DIR /
        "best_model.joblib"
    )

    joblib.dump(
        best_pipeline,
        model_path,
    )

    # ========================================================
    # SAVE MODEL METADATA
    # ========================================================

    metadata_output = {
        "best_model": best_name,

        "roc_auc": float(
            best_score
        ),

        "random_state": int(
            RANDOM_STATE
        ),

        "test_size": float(
            TEST_SIZE
        ),

        "models_evaluated": list(
            models.keys()
        ),

        "selection_metric": "roc_auc",
    }

    metadata_path = (
        MODELS_DIR /
        "model_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata_output,
            file,
            indent=4,
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best model: "
        f"{best_name}"
    )

    print(
        f"ROC-AUC: "
        f"{best_score:.4f}"
    )

    print(
        f"\nModel saved to:\n"
        f"{model_path}"
    )

    print(
        f"\nModel metadata saved to:\n"
        f"{metadata_path}"
    )

    print(
        f"\nModel comparison saved to:\n"
        f"{comparison_path}"
    )

    print()
    print(
        results_df.to_string(
            index=False,
        )
    )

    return results_df


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train_models()
