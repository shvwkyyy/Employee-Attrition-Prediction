from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from ..config import Config
from .schemas import validate_input
import logging
from ..utils.logger import setup_logger

# Set up logger
setup_logger()
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model and preprocessor
try:
    model = joblib.load(Config.MODEL_DIR / 'model' / 'model.pkl')
    preprocessor = joblib.load(Config.MODEL_DIR / 'preprocessor.joblib')
    logger.info("Model and preprocessor loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction on employee attrition"""
    try:
        # Get and validate input data
        input_data = request.get_json()
        errors = validate_input(input_data)
        
        if errors:
            return jsonify({"errors": errors}), 400
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocess
        processed_data = preprocessor.transform(input_df)
        
        # Predict
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)[0][1]
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(probability),
            'message': 'Prediction successful'
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Prediction failed'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)