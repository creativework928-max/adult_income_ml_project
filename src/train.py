"""
Production-oriented model training pipeline.

Trains multiple classification models using the same
leakage-safe preprocessing strategy.

Models:
    - Logistic Regression
    - Random Forest
    - HistGradientBoosting

The best model is selected using ROC-AUC.
"""

import json
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
    RANDOM_STATE,
)

from src.data_loader import (
    load_adult_dataset,
    combine_features_target,
)

from src.preprocessing import (
    build_preprocessing_pipeline,
    clean_dataframe,
    split_dataset,
)


# ============================================================
# MODELS
# ============================================================

def get_models():
    """
    Return the models used for comparison.

    Returns
    -------
    dict
        Dictionary containing model names and estimators.
    """

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE,
        ),
    }


# ============================================================
# SPARSE -> DENSE TRANSFORMER
# ============================================================

def sparse_to_dense(X):
    """
    Convert sparse matrices to dense numpy arrays.

    HistGradientBoostingClassifier requires dense input.

    Parameters
    ----------
    X : array-like or scipy sparse matrix
        Input data.

    Returns
    -------
    numpy.ndarray
        Dense numerical array.
    """

    if sp.issparse(X):
        return X.toarray()

    return np.asarray(X)


# ============================================================
# BUILD MODEL PIPELINE
# ============================================================

def build_model_pipeline(
    model_name,
    model,
):
    """
    Build a model-specific pipeline.

    Logistic Regression and Random Forest can consume
    sparse matrices.

    HistGradientBoosting requires dense input, so its
    preprocessing output is converted to dense.

    Parameters
    ----------
    model_name : str
        Name of the model.

    model : sklearn estimator
        Model estimator.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Complete preprocessing + model pipeline.
    """

    steps = [
        (
            "preprocessor",
            build_preprocessing_pipeline(),
        )
    ]

    # --------------------------------------------------------
    # HistGradientBoosting requires dense input
    # --------------------------------------------------------

    if model_name == "HistGradientBoosting":

        steps.append(
            (
                "to_dense",
                FunctionTransformer(
                    sparse_to_dense,
                    accept_sparse=True,
                ),
            )
        )

    # --------------------------------------------------------
    # Add model
    # --------------------------------------------------------

    steps.append(
        (
            "model",
            model,
        )
    )

    return Pipeline(
        steps=steps,
    )


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models():
    """
    Train and compare all configured models.

    The preprocessing pipeline is fitted separately for each
    model using only the training data.

    Returns
    -------
    pandas.DataFrame
        Model comparison results.
    """

    print("=" * 80)
    print("MODEL TRAINING")
    print("=" * 80)

    # --------------------------------------------------------
    # Create required directories
    # --------------------------------------------------------

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading Adult dataset...")

    X, y, metadata = load_adult_dataset()

    # --------------------------------------------------------
    # Combine features and target
    # --------------------------------------------------------

    df = combine_features_target(
        X,
        y,
    )

    # --------------------------------------------------------
    # Clean dataset
    # --------------------------------------------------------

    df = clean_dataframe(
        df,
    )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    print("Creating train/test split...")

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

    print()

    # --------------------------------------------------------
    # Get models
    # --------------------------------------------------------

    models = get_models()

    results = []

    best_pipeline = None
    best_score = -np.inf
    best_name = None

    # --------------------------------------------------------
    # Train each model
    # --------------------------------------------------------

    for name, model in models.items():

        print("-" * 80)
        print(f"Training: {name}")
        print("-" * 80)

        # ----------------------------------------------------
        # Build model pipeline
        # ----------------------------------------------------

        pipeline = build_model_pipeline(
            model_name=name,
            model=model,
        )

        # ----------------------------------------------------
        # Fit ONLY on training data
        # ----------------------------------------------------

        pipeline.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Predict probabilities
        # ----------------------------------------------------

        probabilities = (
            pipeline.predict_proba(
                X_test,
            )[:, 1]
        )

        # ----------------------------------------------------
        # Convert probabilities to class predictions
        # ----------------------------------------------------

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

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

        results.append(
            metrics
        )

        # ----------------------------------------------------
        # Display model results
        # ----------------------------------------------------

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
            f"F1 Score : "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC  : "
            f"{metrics['roc_auc']:.4f}"
        )

        # ----------------------------------------------------
        # Select best model using ROC-AUC
        # ----------------------------------------------------

        if metrics["roc_auc"] > best_score:

            best_score = metrics["roc_auc"]

            best_pipeline = pipeline

            best_name = name

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    results_df = pd.DataFrame(
        results,
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

    best_model_path = (
        MODELS_DIR /
        "best_model.joblib"
    )

    joblib.dump(
        best_pipeline,
        best_model_path,
    )

    # ========================================================
    # SAVE MODEL METADATA
    # ========================================================

    metadata_output = {
        "best_model": best_name,

        "roc_auc": float(
            best_score
        ),

        "random_state": RANDOM_STATE,

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

    print("\n")
    print("=" * 80)
    print("MODEL TRAINING COMPLETE")
    print("=" * 80)

    print("\nModel comparison:")

    print(
        results_df.to_string(
            index=False,
        )
    )

    print(
        f"\nBest model: "
        f"{best_name}"
    )

    print(
        f"Best ROC-AUC: "
        f"{best_score:.4f}"
    )

    print(
        f"\nModel saved to:"
        f"\n{best_model_path}"
    )

    print(
        f"\nComparison report saved to:"
        f"\n{comparison_path}"
    )

    print(
        f"\nModel metadata saved to:"
        f"\n{metadata_path}"
    )

    return results_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train_models()
