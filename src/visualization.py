"""
Professional visualization module for Task 1.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    FIGURES_DIR,
)


# ============================================================
# PROFESSIONAL COLOR PALETTE
# ============================================================

COLORS = {
    "navy": "#0F172A",
    "blue": "#2563EB",
    "cyan": "#0891B2",
    "green": "#059669",
    "orange": "#D97706",
    "red": "#DC2626",
    "purple": "#7C3AED",
    "slate": "#64748B",
    "light": "#F8FAFC",
    "border": "#CBD5E1",
    "white": "#FFFFFF",
}


# ============================================================
# THEME
# ============================================================

def setup_theme():

    sns.set_theme(
        style="whitegrid",
        context="notebook"
    )

    plt.rcParams.update({

        "figure.dpi": 130,

        "savefig.dpi": 180,

        "figure.facecolor": "white",

        "axes.facecolor": "white",

        "axes.edgecolor": COLORS["border"],

        "axes.labelcolor": COLORS["navy"],

        "axes.titlecolor": COLORS["navy"],

        "axes.titlesize": 17,

        "axes.titleweight": "bold",

        "axes.labelsize": 12,

        "xtick.color": COLORS["slate"],

        "ytick.color": COLORS["slate"],

        "font.family": "DejaVu Sans",
    })


# ============================================================
# SAVE FIGURE
# ============================================================

def save_figure(
    filename,
    output_dir=FIGURES_DIR
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    path = (
        output_dir /
        filename
    )

    plt.savefig(
        path,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    return path


# ============================================================
# MISSING VALUES
# ============================================================

def plot_missing_values(
    df,
    output_dir=FIGURES_DIR
):

    missing = (
        df.isnull()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing = missing[
        missing > 0
    ]

    if missing.empty:
        return None

    plt.figure(
        figsize=(11, 6)
    )

    ax = sns.barplot(
        x=missing.values,
        y=missing.index,
        color=COLORS["blue"]
    )

    ax.set_title(
        "Missing Values by Feature",
        loc="left",
        pad=15
    )

    ax.set_xlabel(
        "Number of Missing Records"
    )

    ax.set_ylabel(
        "Feature"
    )

    for container in ax.containers:

        ax.bar_label(
            container,
            fmt="%.0f",
            padding=4
        )

    sns.despine(
        left=False,
        bottom=False
    )

    plt.tight_layout()

    return save_figure(
        "01_missing_values.png",
        output_dir
    )


# ============================================================
# MISSING VALUE PERCENTAGE
# ============================================================

def plot_missing_percentage(
    df,
    output_dir=FIGURES_DIR
):

    percentage = (
        df.isnull()
        .mean()
        .mul(100)
        .sort_values(
            ascending=False
        )
    )

    percentage = percentage[
        percentage > 0
    ]

    if percentage.empty:
        return None

    plt.figure(
        figsize=(11, 6)
    )

    ax = sns.barplot(
        x=percentage.values,
        y=percentage.index,
        color=COLORS["orange"]
    )

    ax.set_title(
        "Missing Data Percentage",
        loc="left",
        pad=15
    )

    ax.set_xlabel(
        "Missing Values (%)"
    )

    ax.set_ylabel(
        "Feature"
    )

    for container in ax.containers:

        ax.bar_label(
            container,
            fmt="%.2f%%",
            padding=4
        )

    plt.tight_layout()

    return save_figure(
        "02_missing_percentage.png",
        output_dir
    )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def plot_target_distribution(
    df,
    output_dir=FIGURES_DIR
):

    plt.figure(
        figsize=(10, 6)
    )

    order = (
        df["income"]
        .value_counts()
        .index
    )

    ax = sns.countplot(
        data=df,
        x="income",
        order=order,
        palette=[
            COLORS["blue"],
            COLORS["green"]
        ]
    )

    ax.set_title(
        "Income Target Distribution",
        loc="left",
        pad=15
    )

    ax.set_xlabel(
        "Income Class"
    )

    ax.set_ylabel(
        "Number of Records"
    )

    for container in ax.containers:

        ax.bar_label(
            container,
            padding=4
        )

    plt.tight_layout()

    return save_figure(
        "03_target_distribution.png",
        output_dir
    )


# ============================================================
# NUMERICAL DISTRIBUTIONS
# ============================================================

def plot_numeric_distributions(
    df,
    numeric_columns,
    output_dir=FIGURES_DIR
):

    paths = []

    for column in numeric_columns:

        plt.figure(
            figsize=(11, 6)
        )

        sns.histplot(
            data=df,
            x=column,
            kde=True,
            color=COLORS["blue"],
            edgecolor="white",
            alpha=0.85
        )

        plt.title(
            f"Distribution of {column}",
            loc="left",
            pad=15
        )

        plt.xlabel(column)

        plt.ylabel(
            "Frequency"
        )

        plt.tight_layout()

        filename = (
            "04_distribution_"
            f"{column.replace('-', '_')}.png"
        )

        paths.append(
            save_figure(
                filename,
                output_dir
            )
        )

    return paths


# ============================================================
# BOXPLOTS
# ============================================================

def plot_outliers(
    df,
    numeric_columns,
    output_dir=FIGURES_DIR
):

    paths = []

    for column in numeric_columns:

        plt.figure(
            figsize=(11, 4.5)
        )

        sns.boxplot(
            x=df[column],
            color=COLORS["cyan"],
            width=0.45,
            flierprops={
                "marker": "o",
                "markerfacecolor": COLORS["red"],
                "markersize": 4,
                "alpha": 0.35,
            }
        )

        plt.title(
            f"Outlier Analysis — {column}",
            loc="left",
            pad=15
        )

        plt.xlabel(
            column
        )

        plt.tight_layout()

        filename = (
            "05_boxplot_"
            f"{column.replace('-', '_')}.png"
        )

        paths.append(
            save_figure(
                filename,
                output_dir
            )
        )

    return paths


# ============================================================
# CATEGORICAL DISTRIBUTIONS
# ============================================================

def plot_categorical_distribution(
    df,
    categorical_columns,
    output_dir=FIGURES_DIR
):

    paths = []

    for column in categorical_columns:

        counts = (
            df[column]
            .fillna("Missing")
            .value_counts()
            .head(15)
            .sort_values()
        )

        plt.figure(
            figsize=(12, 7)
        )

        ax = sns.barplot(
            x=counts.values,
            y=counts.index,
            color=COLORS["purple"]
        )

        ax.set_title(
            f"Top Categories — {column}",
            loc="left",
            pad=15
        )

        ax.set_xlabel(
            "Number of Records"
        )

        ax.set_ylabel(
            column
        )

        for container in ax.containers:

            ax.bar_label(
                container,
                padding=3,
                fontsize=8
            )

        plt.tight_layout()

        filename = (
            "06_category_"
            f"{column.replace('-', '_')}.png"
        )

        paths.append(
            save_figure(
                filename,
                output_dir
            )
        )

    return paths


# ============================================================
# CORRELATION
# ============================================================

def plot_correlation(
    df,
    numeric_columns,
    output_dir=FIGURES_DIR
):

    correlation = (
        df[numeric_columns]
        .corr()
    )

    plt.figure(
        figsize=(11, 8)
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.7,
        square=True,
        cbar_kws={
            "label": "Correlation"
        }
    )

    plt.title(
        "Numerical Feature Correlation Matrix",
        loc="left",
        pad=15
    )

    plt.tight_layout()

    return save_figure(
        "07_correlation_heatmap.png",
        output_dir
    )


# ============================================================
# COMPLETE VISUALIZATION RUNNER
# ============================================================

def generate_all_visualizations(
    df,
    numeric_columns,
    categorical_columns,
    output_dir=FIGURES_DIR
):

    setup_theme()

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    generated = []

    functions = [

        lambda: plot_missing_values(
            df,
            output_dir
        ),

        lambda: plot_missing_percentage(
            df,
            output_dir
        ),

        lambda: plot_target_distribution(
            df,
            output_dir
        ),

        lambda: plot_numeric_distributions(
            df,
            numeric_columns,
            output_dir
        ),

        lambda: plot_outliers(
            df,
            numeric_columns,
            output_dir
        ),

        lambda: plot_categorical_distribution(
            df,
            categorical_columns,
            output_dir
        ),

        lambda: plot_correlation(
            df,
            numeric_columns,
            output_dir
        ),
    ]

    for function in functions:

        result = function()

        if result is None:
            continue

        if isinstance(
            result,
            list
        ):

            generated.extend(
                result
            )

        else:

            generated.append(
                result
            )

    return generated