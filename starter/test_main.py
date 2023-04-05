"""
Script with the automatic tests for the app
"""
import json
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {
        "greeting": "Welcome to Udacity Project: Deploying a ML Model to Cloud Application Platform with FastAPI, author Jose Lopez"}


def test_post_over_50k():
    data = {
        "age": 52,
        "workclass": "Self-emp-inc",
        "fnlgt": 287927,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    r = client.post("/model_inference/", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json() == {'Prediction: ': 'Salary is >50K'}


def test_post_less_50k():
    data = {"age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White", "sex": "Male",
            "capital-gain": 2174, "capital-loss": 0,
            "hours-per-week": 40, "native-country":
            "United-States"
            }
    r = client.post("/model_inference/", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json() == {'Prediction: ': 'Salary is <=50K'}
