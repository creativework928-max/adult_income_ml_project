import pandas as pd
import pytest

from src.validation import (
    validate_schema,
)


def test_schema_validation():

    df = pd.DataFrame({
        "age": [30],
        "fnlwgt": [100000],
        "education-num": [13],
        "capital-gain": [0],
        "capital-loss": [0],
        "hours-per-week": [40],
        "workclass": ["Private"],
        "education": ["Bachelors"],
        "marital-status":
            ["Never-married"],
        "occupation":
            ["Prof-specialty"],
        "relationship":
            ["Not-in-family"],
        "race": ["White"],
        "sex": ["Male"],
        "native-country":
            ["United-States"],
        "income": [">50K"],
    })

    assert validate_schema(df)


def test_invalid_schema():

    df = pd.DataFrame({
        "age": [30]
    })

    with pytest.raises(
        ValueError
    ):

        validate_schema(df)