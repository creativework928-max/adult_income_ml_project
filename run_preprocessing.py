"""
Main execution script for Task 1:
Data Preprocessing.
"""

import traceback

import pandas as pd
from scipy import sparse

from src.config import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    SPLIT_DATA_DIR,
    LOGS_DIR,
    REPORTS_DIR,
    TRAIN_RAW_FILE,
    TEST_RAW_FILE,
    TRAIN_PROCESSED_FILE,
    TEST_PROCESSED_FILE,
    Y_TRAIN_FILE,
    Y_TEST_FILE,
    CLEANED_DATA_FILE,
    PREPROCESSING_SUMMARY_FILE,
)

from src.data_loader import (
    load_and_prepare_raw_data,
    get_dataset_summary,
)

from src.preprocessing import (
    clean_dataframe,
    preprocess_data,
    get_processed_feature_names,
)

from src.visualization import (
    generate_all_visualizations,
)

from src.utils import (
    setup_logger,
    set_random_seed,
    print_banner,
)


# ============================================================
# LOGGER
# ============================================================

logger = setup_logger(
    log_file=(
        LOGS_DIR /
        "preprocessing.log"
    )
)


# ============================================================
# MAIN
# ============================================================

def main():

    print_banner(
        "TASK 1 — DATA PREPROCESSING"
    )

    set_random_seed(42)

    try:

        # ----------------------------------------------------
        # STEP 1 — LOAD DATA
        # ----------------------------------------------------

        logger.info(
            "Step 1/7 — Loading official UCI dataset."
        )

        df, metadata = (
            load_and_prepare_raw_data()
        )

        summary = (
            get_dataset_summary(df)
        )

        logger.info(
            f"Dataset loaded: "
            f"{summary['rows']:,} rows × "
            f"{summary['columns']} columns."
        )

        # ----------------------------------------------------
        # STEP 2 — CLEAN DATA
        # ----------------------------------------------------

        logger.info(
            "Step 2/7 — Cleaning dataset."
        )

        df_clean = clean_dataframe(
            df
        )

        df_clean.to_csv(
            CLEANED_DATA_FILE,
            index=False
        )

        # ----------------------------------------------------
        # STEP 3 — VISUALIZATION
        # ----------------------------------------------------

        logger.info(
            "Step 3/7 — Generating visualizations."
        )

        from src.config import (
            NUMERIC_FEATURES,
            CATEGORICAL_FEATURES,
        )

        generated_figures = (
            generate_all_visualizations(
                df_clean,
                NUMERIC_FEATURES,
                CATEGORICAL_FEATURES,
                FIGURES_DIR
            )
        )

        logger.info(
            f"Generated "
            f"{len(generated_figures)} "
            f"visualization files."
        )

        # ----------------------------------------------------
        # STEP 4 — PREPROCESS
        # ----------------------------------------------------

        logger.info(
            "Step 4/7 — Fitting preprocessing pipeline."
        )

        (
            X_train,
            X_test,
            y_train,
            y_test,
            X_train_processed,
            X_test_processed,
            preprocessor
        ) = preprocess_data(
            df_clean
        )

        # ----------------------------------------------------
        # STEP 5 — SAVE RAW SPLITS
        # ----------------------------------------------------

        logger.info(
            "Step 5/7 — Saving train/test splits."
        )

        train_raw = (
            X_train.copy()
        )

        train_raw["income"] = (
            y_train.values
        )

        test_raw = (
            X_test.copy()
        )

        test_raw["income"] = (
            y_test.values
        )

        train_raw.to_csv(
            TRAIN_RAW_FILE,
            index=False
        )

        test_raw.to_csv(
            TEST_RAW_FILE,
            index=False
        )

        # ----------------------------------------------------
        # STEP 6 — SAVE PROCESSED DATA
        # ----------------------------------------------------

        logger.info(
            "Step 6/7 — Saving processed matrices."
        )

        sparse.save_npz(
            TRAIN_PROCESSED_FILE,
            X_train_processed
        )

        sparse.save_npz(
            TEST_PROCESSED_FILE,
            X_test_processed
        )

        pd.DataFrame({
            "income": y_train
        }).to_csv(
            Y_TRAIN_FILE,
            index=False
        )

        pd.DataFrame({
            "income": y_test
        }).to_csv(
            Y_TEST_FILE,
            index=False
        )

        # ----------------------------------------------------
        # STEP 7 — REPORT
        # ----------------------------------------------------

        logger.info(
            "Step 7/7 — Creating preprocessing report."
        )

        feature_names = (
            get_processed_feature_names(
                preprocessor
            )
        )

        report = pd.DataFrame({

            "Metric": [

                "Original rows",

                "Original columns",

                "Training rows",

                "Testing rows",

                "Original numerical features",

                "Original categorical features",

                "Processed features",

                "Missing cells before preprocessing",

                "Missing cells after deterministic cleaning",

                "Training positive-class ratio",

                "Testing positive-class ratio",

                "Train/Test split",

                "Random state",

            ],

            "Value": [

                len(df),

                len(df.columns),

                len(X_train),

                len(X_test),

                len(NUMERIC_FEATURES),

                len(CATEGORICAL_FEATURES),

                len(feature_names),

                int(
                    df.isnull()
                    .sum()
                    .sum()
                ),

                int(
                    df_clean.isnull()
                    .sum()
                    .sum()
                ),

                round(
                    y_train.mean(),
                    4
                ),

                round(
                    y_test.mean(),
                    4
                ),

                "80% / 20%",

                42,
            ]
        })

        report.to_csv(
            PREPROCESSING_SUMMARY_FILE,
            index=False
        )

        # ----------------------------------------------------
        # FINAL MESSAGE
        # ----------------------------------------------------

        print_banner(
            "TASK 1 COMPLETED SUCCESSFULLY"
        )

        print(
            f"\nDataset:\n"
            f"  {len(df):,} rows"
        )

        print(
            f"\nTrain/Test:\n"
            f"  {len(X_train):,} / "
            f"{len(X_test):,}"
        )

        print(
            f"\nProcessed Features:\n"
            f"  {len(feature_names):,}"
        )

        print(
            "\nGenerated outputs:"
        )

        print(
            f"  ✓ Raw data\n"
            f"  ✓ Cleaned data\n"
            f"  ✓ Train/test split\n"
            f"  ✓ Processed matrices\n"
            f"  ✓ Preprocessing pipeline\n"
            f"  ✓ Professional visualizations\n"
            f"  ✓ Preprocessing summary\n"
            f"  ✓ Execution log"
        )

        logger.info(
            "Task 1 completed successfully."
        )

    except Exception as error:

        logger.exception(
            "Task 1 failed."
        )

        print(
            "\nERROR:"
        )

        print(
            str(error)
        )

        traceback.print_exc()

        raise


if __name__ == "__main__":
    main()