"""
General project utilities.
"""

import json
import logging
import random

from datetime import datetime
from pathlib import Path

import numpy as np


# ============================================================
# RANDOM SEED
# ============================================================

def set_random_seed(seed=42):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)


# ============================================================
# DIRECTORY CREATION
# ============================================================

def ensure_directory(path):
    """
    Create directory if it does not exist.
    """

    path = Path(path)

    path.mkdir(
        parents=True,
        exist_ok=True
    )

    return path


# ============================================================
# LOGGING
# ============================================================

def setup_logger(
    name="adult_income_project",
    log_file=None
):
    """
    Configure professional console/file logging.
    """

    logger = logging.getLogger(name)

    logger.setLevel(
        logging.INFO
    )

    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        console_handler
    )

    if log_file is not None:

        log_file = Path(log_file)

        ensure_directory(
            log_file.parent
        )

        file_handler = logging.FileHandler(
            log_file,
            encoding="utf-8"
        )

        file_handler.setFormatter(
            formatter
        )

        logger.addHandler(
            file_handler
        )

    return logger


# ============================================================
# JSON SAVING
# ============================================================

def save_json(data, path):
    """
    Save dictionary/object as JSON.
    """

    path = Path(path)

    ensure_directory(
        path.parent
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            default=str
        )


# ============================================================
# TIMESTAMP
# ============================================================

def current_timestamp():
    """
    Return formatted current timestamp.
    """

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ============================================================
# FILE SIZE
# ============================================================

def get_file_size_mb(path):
    """
    Return file size in megabytes.
    """

    path = Path(path)

    if not path.exists():
        return 0.0

    return round(
        path.stat().st_size /
        (1024 ** 2),
        2
    )


# ============================================================
# SAFE PERCENTAGE
# ============================================================

def percentage(part, total):
    """
    Calculate percentage safely.
    """

    if total == 0:
        return 0.0

    return round(
        (part / total) * 100,
        2
    )


# ============================================================
# PROJECT BANNER
# ============================================================

def print_banner(title):
    """
    Print a professional terminal banner.
    """

    width = 80

    print("\n")
    print("=" * width)
    print(title.center(width))
    print("=" * width)