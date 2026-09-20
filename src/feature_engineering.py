import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # --------------------------------------------------------
    # Capital-gain / loss indicator
    # --------------------------------------------------------

    if "capital-gain" in df.columns:

        df["has_capital_gain"] = (
            df["capital-gain"] > 0
        ).astype(int)

    if "capital-loss" in df.columns:

        df["has_capital_loss"] = (
            df["capital-loss"] > 0
        ).astype(int)

    # --------------------------------------------------------
    # Total capital movement
    # --------------------------------------------------------

    if {
        "capital-gain",
        "capital-loss"
    }.issubset(df.columns):

        df["net_capital"] = (
            df["capital-gain"] -
            df["capital-loss"]
        )

    # --------------------------------------------------------
    # Work intensity
    # --------------------------------------------------------

    if {
        "age",
        "hours-per-week"
    }.issubset(df.columns):

        df["hours_per_age"] = (
            df["hours-per-week"] /
            df["age"].replace(0, 1)
        )

    return df