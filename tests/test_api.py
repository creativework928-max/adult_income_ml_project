from fastapi.testclient import (
    TestClient
)

from api.main import app


client = TestClient(app)


def test_root():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json()["status"] == "running"


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"


def test_prediction():

    payload = {
        "age": 35,
        "workclass": "Private",
        "fnlwgt": 180000,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status":
            "Married-civ-spouse",
        "occupation":
            "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country":
            "United-States",
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    body = response.json()

    assert "prediction" in body
    assert "label" in body
    assert "probability" in body

    assert body["prediction"] in [0, 1]

    assert 0 <= body["probability"] <= 1