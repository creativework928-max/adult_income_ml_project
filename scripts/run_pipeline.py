from src.data_loader import (
    load_adult_dataset,
    combine_features_target,
)

from src.preprocessing import (
    clean_dataframe,
    preprocess_data,
)

from src.train import (
    train_models,
)

from src.evaluate import (
    evaluate_model,
)

from src.explainability import (
    extract_feature_importance,
)

from src.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FIGURES_DIR,
)

from src.visualization import (
    generate_all_visualizations,
)


def main():

    print("=" * 80)
    print("END-TO-END ML PIPELINE")
    print("=" * 80)

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print("\n[1] Loading data...")

    X, y, metadata = (
        load_adult_dataset()
    )

    df = combine_features_target(
        X,
        y
    )

    df = clean_dataframe(df)

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    print(
        "\n[2] Generating visualizations..."
    )

    generate_all_visualizations(
        df,
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        FIGURES_DIR,
    )

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    print(
        "\n[3] Preprocessing..."
    )

    preprocess_data(df)

    # --------------------------------------------------------
    # MODEL TRAINING
    # --------------------------------------------------------

    print(
        "\n[4] Training models..."
    )

    train_models()

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    print(
        "\n[5] Evaluating model..."
    )

    evaluate_model()

    # --------------------------------------------------------
    # EXPLAINABILITY
    # --------------------------------------------------------

    print(
        "\n[6] Generating feature importance..."
    )

    extract_feature_importance()

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()