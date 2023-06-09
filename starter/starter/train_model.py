"""
Script that trains an inference model and saves it as a .pkl file. It also stores the related enconders
"""
from sklearn.model_selection import train_test_split
import joblib
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, compute_model_metrics_slice
import pandas as pd
import os


# Add code to load in the data.

# print(os.path.realpath(__file__))
file_path = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(file_path, "..", "data", "census.csv"))

# Some data cleaning
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
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


joblib.dump(encoder, os.path.join(file_path, "..", "model", "encoder.pkl"))
joblib.dump(lb, os.path.join(file_path, "..", "model", "lb.pkl"))

encoder = joblib.load(os.path.join(file_path, "..", "model", "encoder.pkl"))
lb = joblib.load(os.path.join(file_path, "..", "model", "lb.pkl"))

# Proces the test data with the process_data function.
X_test, y_test, __, __ = process_data(
    test, categorical_features=cat_features, label='salary', training=False, encoder=encoder, lb=lb
)

# Train and save a model.
rf = train_model(X_train, y_train)

pred = inference(rf, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, pred)
print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"fbeta: {fbeta}")

compute_model_metrics_slice(test, y_test, pred, cat_features)
