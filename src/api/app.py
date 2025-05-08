from flask import Flask, request, jsonify
import logging
#local imports
from src.api.predict import predictor
# from src.api.schemas import EmployeeData

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        input_data = request.get_json()
        result = predictor.predict(input_data)
        return jsonify(result)
    except ValueError as e:
        return jsonify(e.args[0]), 400
    except Exception as e:
        return jsonify({
            'error': 'Prediction service unavailable',
            'message': str(e)
        }), 503

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'ready': predictor.model is not None and predictor.preprocessor is not None
    }
    return jsonify(status), 200 if status['ready'] else 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)