"""
Central configuration for the Adult Income ML project.
"""

from pathlib import Path
import os

from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# DIRECTORY STRUCTURE
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLIT_DATA_DIR = DATA_DIR / "splits"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

SRC_DIR = PROJECT_ROOT / "src"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
LOGS_DIR = OUTPUT_DIR / "logs"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

MODELS_DIR = PROJECT_ROOT / "models"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

DIRECTORIES = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SPLIT_DATA_DIR,
    NOTEBOOKS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    PREDICTIONS_DIR,
    MODELS_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASET_NAME = "Adult Census Income"

UCI_DATASET_ID = 2

TARGET_COLUMN = "income"


# ============================================================
# RANDOMNESS / SPLITTING
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ============================================================
# NUMERICAL FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


# ============================================================
# OUTLIER CONFIGURATION
# ============================================================

OUTLIER_LOWER_QUANTILE = 0.01

OUTLIER_UPPER_QUANTILE = 0.99


# ============================================================
# FILE NAMES
# ============================================================

RAW_DATA_FILE = (
    RAW_DATA_DIR /
    "adult_raw.csv"
)

CLEANED_DATA_FILE = (
    PROCESSED_DATA_DIR /
    "cleaned_dataset.csv"
)

PREPROCESSING_SUMMARY_FILE = (
    PROCESSED_DATA_DIR /
    "preprocessing_summary.csv"
)

PREPROCESSING_REPORT_FILE = (
    REPORTS_DIR /
    "preprocessing_report.html"
)

PIPELINE_FILE = (
    MODELS_DIR /
    "preprocessing_pipeline.joblib"
)

TRAIN_RAW_FILE = (
    SPLIT_DATA_DIR /
    "train_raw.csv"
)

TEST_RAW_FILE = (
    SPLIT_DATA_DIR /
    "test_raw.csv"
)

TRAIN_PROCESSED_FILE = (
    SPLIT_DATA_DIR /
    "X_train_processed.npz"
)

TEST_PROCESSED_FILE = (
    SPLIT_DATA_DIR /
    "X_test_processed.npz"
)

Y_TRAIN_FILE = (
    SPLIT_DATA_DIR /
    "y_train.csv"
)

Y_TEST_FILE = (
    SPLIT_DATA_DIR /
    "y_test.csv"
)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        MODELS_DIR / "best_model.joblib"
    )
)

PREPROCESSOR_PATH = Path(
    os.getenv(
        "PREPROCESSOR_PATH",
        MODELS_DIR / "preprocessing_pipeline.joblib"
    )

)


APP_NAME = os.getenv(
    "APP_NAME",
    "Adult Income Prediction API"
)

APP_VERSION = os.getenv(
    "APP_VERSION",
    "1.0.0"
)

API_HOST = os.getenv(
    "API_HOST",
    "0.0.0.0"
)

API_PORT = int(
    os.getenv(
        "API_PORT",
        "8000"
    )
)