import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Define a Pydantic model for input validation
class PredictionInput(BaseModel):
    features: List[float]  # Adjust based on your model's input requirements

# Load the Pickle model (ensure wifi_threat_model.pkl is in the same directory)
try:
    with open('wifi_threat_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading pickle file: {e}")
    raise

@app.post("/predict")
async def predict(data: PredictionInput):
    try:
        # Extract features and reshape for the model
        features = np.array([data.features]).reshape(1, -1)
        # Make prediction
        prediction = model.predict(features)[0]
        # Return prediction as JSON
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)