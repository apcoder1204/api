from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import cloudpickle

app = FastAPI()

# Load model with cloudpickle
with open("rf_wifi_threat_model.pkl", "rb") as f:
    model = cloudpickle.load(f)

# Define expected input features
class Features(BaseModel):
    Packet_Size: float
    TTL: int
    Time_Delta: float
    src_port: int
    dst_port: int
    Known_BSSID: str
    Protocol: str
    TCP_Flags: str
    Packet_Direction: str

@app.post("/predict")
def predict(data: Features):
    try:
        # Convert input to DataFrame with correct structure
        import pandas as pd

        input_df = pd.DataFrame([data.dict()])
        # One-hot encode categorical columns to match training features
        input_encoded = pd.get_dummies(input_df)

        # Align with model's expected feature order
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in input_encoded:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_features]

        prediction = model.predict(input_encoded)[0]
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
