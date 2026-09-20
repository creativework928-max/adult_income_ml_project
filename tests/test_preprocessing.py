import pandas as pd

from src.preprocessing import (
    clean_dataframe,
    encode_target,
)


def test_question_mark_conversion():

    df = pd.DataFrame({
        "workclass": ["?"]
    })

    cleaned = clean_dataframe(df)

    assert cleaned["workclass"].isna().iloc[0]


def test_target_encoding():

    y = pd.Series([
        "<=50K",
        ">50K"
    ])

    encoded = encode_target(y)

    assert encoded.tolist() == [
        0,
        1
    ]