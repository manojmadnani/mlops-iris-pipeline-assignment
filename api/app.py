# api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import logging
import pandas as pd

app = FastAPI()
model = joblib.load("models/RandomForest.pkl")

# Logger
logging.basicConfig(filename="logs/api.log", level=logging.INFO)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Mapping input fields to trained column names
feature_mapping = {
    "sepal_length": "sepal length (cm)",
    "sepal_width": "sepal width (cm)",
    "petal_length": "petal length (cm)",
    "petal_width": "petal width (cm)"
}

@app.post("/predict")
def predict(input: IrisInput):
    input_dict = input.dict()
    
    # Rename keys to match training columns
    mapped_input = {
        feature_mapping[k]: v for k, v in input_dict.items()
    }
    
    df = pd.DataFrame([mapped_input])  # Wrap as DataFrame with proper names
    prediction = model.predict(df)[0]
    logging.info(f"Input: {input_dict} | Prediction: {prediction}")
    return {"prediction": int(prediction)}
