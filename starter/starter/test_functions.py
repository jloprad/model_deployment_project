import pandas as pd
import pytest
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from ml.data import process_data
from ml.model import *

@pytest.fixture
def cat_features():
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        ]
    return cat_features

@pytest.fixture
def data(cat_features):
    """ Function that loads the data from a .csv and prepare the train an test groups"""

    df = pd.read_csv('../data/census.csv')
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()

    train, test = train_test_split(df, test_size=0.20, random_state=24)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, __, __ = process_data(
        test, categorical_features=cat_features,label='salary', training=False, encoder=encoder, lb=lb
    )
    return train, test, X_train, y_train, encoder, lb ,X_test, y_test


def test_train_model( data):
    """ Test if the model is the correct type of class"""
    train, test,X_train, y_train, encoder, lb ,X_test, y_test = data

    os.remove('../model/rf_model.pkl')
    rf = train_model(X_train, y_train)
    assert isinstance(rf, RandomForestClassifier)


def test_train_model_file():
    """ Test if the model file was created """
    assert os.path.isfile('../model/rf_model.pkl')

def test_inference( data):
    """ Test that the inference function is returning values"""

    train, test,X_train, y_train, encoder, lb ,X_test, y_test = data
    rf = joblib.load('../model/rf_model.pkl')
    pred = inference(rf, X_test)

    assert  len(pred)>0

def test_compute_model_metrics( data):
    """ Test that the model metrics function is returning values within the expected range"""

    train, test,X_train, y_train, encoder, lb ,X_test, y_test = data
    rf = joblib.load('../model/rf_model.pkl')
    pred = inference(rf, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, pred)

    assert (0 <= precision <= 1).all()
    assert (0 <= recall <= 1).all()
    assert (0 <= fbeta <= 1).all()

def test_compute_model_metrics_slice( data, cat_features):
    """ Test that the file with the model metrics is created """

    train, test,X_train, y_train, encoder, lb ,X_test, y_test = data
    rf = joblib.load('../model/rf_model.pkl')
    pred = inference(rf, X_test)

    os.remove('../model/slice_output.txt')
    compute_model_metrics_slice(test, y_test, pred, cat_features)
    assert os.path.isfile('../model/slice_output.txt')
