# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
import joblib
from ml.data import process_data ########### PATH
from ml.model import * ########### PATH
import pandas as pd


# Add code to load in the data.

df = pd.read_csv('../data/census.csv')

# Some data cleaning
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(df, test_size=0.20, random_state=24)

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


X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, __, __ = process_data(
    test, categorical_features=cat_features,label='salary', training=False, encoder=encoder, lb=lb
)

# Train and save a model.
rf = train_model(X_train, y_train)

pred = inference(rf, X_test)
compute_model_metrics_slice(test, y_test, pred)

joblib.dump( rf, '../model/rf_model.pkl')
