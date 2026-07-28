"""
Flask Backend for SuperKart Sales Forecasting Model
"""
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

try:
    model = joblib.load('random_forest_model.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    print("Model and preprocessor loaded successfully")
except FileNotFoundError:
    print("Model files not found.")
    model = None
    preprocessor = None

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'API is running', 'model': 'SuperKart Sales Forecasting', 'version': '1.0'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        input_df = pd.DataFrame([data])
        if preprocessor is not None:
            X_processed = preprocessor.transform(input_df)
        else:
            return jsonify({'error': 'Preprocessor not available'}), 500
        if model is not None:
            prediction = model.predict(X_processed)[0]
            return jsonify({'prediction': float(prediction), 'input_data': data, 'status': 'success'}), 200
        else:
            return jsonify({'error': 'Model not available'}), 500
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 400

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        records = data.get('records', [])
        if not records:
            return jsonify({'error': 'No records provided'}), 400
        input_df = pd.DataFrame(records)
        if preprocessor is not None:
            X_processed = preprocessor.transform(input_df)
        else:
            return jsonify({'error': 'Preprocessor not available'}), 500
        if model is not None:
            predictions = model.predict(X_processed)
            return jsonify({'predictions': [float(p) for p in predictions], 'num_records': len(records), 'status': 'success'}), 200
        else:
            return jsonify({'error': 'Model not available'}), 500
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None, 'preprocessor_loaded': preprocessor is not None}), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)