# Model Card

## Model Details
- Author: Jose Lopez
- Date: April 2023
- Model version: 1.0.0
- Model type: Random forest

## Intended Use
Predicts if the salary of a person is over or under 50K USD based on census data.

## Training Data
Census Income Data Set
https://archive.ics.uci.edu/ml/datasets/census+income
A 80% of the dataset was used for training.
For preprocessing a label binarizer and one hot encoding are used.

## Evaluation Data
A split of 20% of the original dataset that was used for training.

## Metrics
The results on the evaluation data:
_precision: 0.7320359281437125_
_recall: 0.6201648700063411_
_fbeta: 0.6714727085478888_

## Ethical Considerations
The model uses information about race and gender, which is sensitive data and could be considered discrimination in some cases.

## Caveats and Recommendations
This is a very simple model that can be improved, the main focus of the project is the API deployment with CI/CD.
