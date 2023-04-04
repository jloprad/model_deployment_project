from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import os

file_path= os.path.dirname(os.path.abspath(__file__))

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    rfc = RandomForestClassifier(random_state=24)

    param_grid = {
    'n_estimators': [50, 100],
    'max_features': ['sqrt'],
    'max_depth': [4, 5, 100],
    'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    rf = cv_rfc.best_estimator_


    joblib.dump( rf, os.path.join( file_path,"..","..","model", "rf_model.pkl"))

    return rf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    predictions = model.predict(X)
    return predictions

def compute_model_metrics_slice(x, y, preds, cat_features ):
    """
    Calculates the model metrics for all the slices of categorical variables

    Inputs
    ------
    x : DataFrame
        Original dataframe, before one hot encoding
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """

    # Only works for categorical variables
    f = open(os.path.join( file_path,"..","..","model", "slice_output.txt"), "w")
    f.write("Model metrics for each slice \n")
    f = open(os.path.join( file_path,"..","..","model", "slice_output.txt"), "a")

    for cat in cat_features:
        f.write("\n")
        f.write(f"Variable: {cat} \n")
        for value in x[cat].unique():
            f.write("\n")
            f.write(f" Slice: {value} \n")
            mask = (x[cat]==value)
            precision, recall, fbeta = compute_model_metrics( y[mask] , preds[mask] )
            f.write(f"  Precision: {precision} \n")
            f.write(f"  Recall: {recall} \n")
            f.write(f"  Fbeta: {fbeta} \n")

    return None
