from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
from starter.ml.model import *
from starter.ml.data import process_data
import pandas as pd
import os


file_path = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

class model_variables(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")
    class Config:
        allow_population_by_field_name = True

# Get with welcome message
@app.get("/")
async def greeting():
    return {"greeting": "Welcome to Udacity Project: Deploying a ML Model to Cloud Application Platform with FastAPI, author Jose Lopez"}


@app.post("/model_inference/")
async def model_inference(item: model_variables):

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

    #  Transform the input into a pandas dataframe
    df= pd.DataFrame(list(item))
    df2= pd.DataFrame( df[1]  )
    df2.index= df[0].replace('_','-', regex=True)
    df2 = df2.transpose()

    # Load the pipeline
    encoder = joblib.load(os.path.join( file_path,"model", "encoder.pkl"))
    lb = joblib.load(os.path.join( file_path,"model", "lb.pkl"))

    # Enconde the input
    x_inference, __, __, __ = process_data(
        df2, categorical_features=cat_features,label=None, training=False, encoder=encoder, lb=lb
    )

    # Get the prediction
    rf = joblib.load(os.path.join( file_path,"model", "rf_model.pkl"))
    num_pred = inference(rf, x_inference)

    if num_pred == 1:
        pred = 'Salary is >50K'
    elif num_pred == 0:
        pred = 'Salary is <=50K'
    else:
        pred = 'Unexpected output'

    return {'Prediction: ':pred}
