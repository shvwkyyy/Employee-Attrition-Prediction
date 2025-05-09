from flask import Flask, request, jsonify ,send_from_directory
from flask_cors import CORS
#local imports
from src.api.predict import predictor


app = Flask(__name__, static_folder='src/static')
CORS(app)

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

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

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)