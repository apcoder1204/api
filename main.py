# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()
model = pickle.load(open("wifi_threat_model1copy.pkl", "rb"))

class Features(BaseModel):
    features: list

@app.post("/predict")
def predict(data: Features):
    try:
        input_data = np.array(data.features).reshape(1, -1)
        prediction = model.predict(input_data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
