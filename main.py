# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os
# Load the model
model_path = os.path.join(os.path.dirname(__file__), "wifi_threat_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Define input schema
class WiFiFeatures(BaseModel):
    features: list[float]

# Initialize app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "WiFi Threat Detection API"}

@app.post("/predict")
def predict(data: WiFiFeatures):
    try:
        X = np.array(data.features).reshape(1, -1)
        prediction = model.predict(X)
        return {"threat_detected": bool(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
