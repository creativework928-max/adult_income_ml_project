import joblib
import pandas as pd

from src.config import (
    MODELS_DIR,
)


MODEL_PATH = (
    MODELS_DIR /
    "best_model.joblib"
)


class IncomePredictor:

    def __init__(self):

        self.model = joblib.load(
            MODEL_PATH
        )

    def predict(self, data):

        df = pd.DataFrame(
            [data]
        )

        prediction = int(
            self.model.predict(df)[0]
        )

        probability = float(
            self.model.predict_proba(df)[0][1]
        )

        label = (
            ">50K"
            if prediction == 1
            else "<=50K"
        )

        return {
            "prediction": prediction,
            "label": label,
            "probability": probability,
        }