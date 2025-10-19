import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/invocations', methods=['POST'])
def predict():
    """
    Model inference endpoint.
    Accepts JSON with input data and returns predictions.
    """
    try:
        # Get input data from request
        data = request.get_json()

        # Extract inputs (expecting a list or array-like structure)
        if 'inputs' in data:
            inputs = data['inputs']
        elif 'data' in data:
            inputs = data['data']
        else:
            inputs = data

        # Convert to numpy array if needed
        if isinstance(inputs, list):
            inputs = np.array(inputs)

        # Dummy prediction logic - replace with your actual model
        # For now, just return random predictions
        if isinstance(inputs, np.ndarray):
            predictions = ["Fraud" if np.random.rand() > 0.5 else "Safe" for _ in range(100) for arr in inputs]
        else:
            predictions = ["Fraud" if np.random.rand() > 0.5 else "Safe" for _ in range(100)]

        return jsonify({
            'predictions': predictions
        }), 200

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200


@app.route('/ping', methods=['GET'])
def ping():
    """Ping endpoint for MLflow compatibility."""
    return '', 200


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5002, debug=False)
