import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load model data
with open("linear_model.pkl", "rb") as f:
    model_data = pickle.load(f)

theta = model_data["theta"]
mean = model_data["mean"]
std = model_data["std"]

app = FastAPI()

class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Linear Regression API is running"}

@app.post("/predict")
def predict(data: InputData):
    x = np.array(data.features)

    if len(x) != 8 :
        return {"Error : Model expects exactly 8 feauturs"}

    # Scale input
    x_scaled = (x - mean) / std

    # Add bias term
    x_bias = np.insert(x_scaled, 0, 1)

    # Predict
    prediction = np.dot(x_bias, theta)

    return {"prediction": float(prediction.item())}