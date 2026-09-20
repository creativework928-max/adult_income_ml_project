from pydantic import BaseModel, Field


class IncomeRequest(BaseModel):

    age: float = Field(
        ...,
        ge=17,
        le=100
    )

    workclass: str

    fnlwgt: float = Field(
        ...,
        gt=0
    )

    education: str

    education_num: float = Field(
        ...,
        ge=1,
        le=20
    )

    marital_status: str

    occupation: str

    relationship: str

    race: str

    sex: str

    capital_gain: float = Field(
        ...,
        ge=0
    )

    capital_loss: float = Field(
        ...,
        ge=0
    )

    hours_per_week: float = Field(
        ...,
        ge=1,
        le=100
    )

    native_country: str


class PredictionResponse(BaseModel):

    prediction: int

    label: str

    probability: float