from fastapi import (
    FastAPI,
    HTTPException
)

from api.schemas import (
    IncomeRequest,
    PredictionResponse,
)

from src.config import (
    APP_NAME,
    APP_VERSION,
)

from src.inference import (
    IncomePredictor,
)


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Production-style Adult Income "
        "classification API."
    ),
)


predictor = IncomePredictor()


@app.get("/")
def root():

    return {
        "application": APP_NAME,
        "version": APP_VERSION,
        "status": "running",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    request: IncomeRequest
):

    try:

        data = {
            "age": request.age,
            "workclass": request.workclass,
            "fnlwgt": request.fnlwgt,
            "education": request.education,
            "education-num":
                request.education_num,
            "marital-status":
                request.marital_status,
            "occupation":
                request.occupation,
            "relationship":
                request.relationship,
            "race": request.race,
            "sex": request.sex,
            "capital-gain":
                request.capital_gain,
            "capital-loss":
                request.capital_loss,
            "hours-per-week":
                request.hours_per_week,
            "native-country":
                request.native_country,
        }

        result = predictor.predict(
            data
        )

        return result

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )