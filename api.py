import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the Pickle model (ensure wifi_threat_model.pkl is in the same directory)
with open('wifi_threat_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the WiFi security app
        data = request.get_json()
        
        # Extract features (adjust based on your model's input requirements)
        features = np.array([data['features']]).reshape(1, -1)  # Example: expects a list of features
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Return prediction as JSON
        return jsonify({'prediction': int(prediction)})  # Adjust based on your model's output
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)