"""
Script that tests the POST of the API deployed on render
"""
import requests
import json

# With this input the expected output is: 'Salary is >50K'
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

r = requests.post(
    "https://model-deployment-project.onrender.com/model_inference/",
    data=json.dumps(data))

# Print results
print(f"Status code returned: {r.status_code}")
print(r.json())
