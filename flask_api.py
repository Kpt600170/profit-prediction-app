from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained ElasticNet model
MODEL_PATH = "C:/Users/Keshavi/Downloads/New folder (2)/elastic_net_model.pkl"
model = joblib.load(MODEL_PATH) if MODEL_PATH else None

@app.route('/predict_excel', methods=['POST'])
def predict_from_excel():
    """Predict profits from an uploaded Excel file."""
    if model is None:
        return jsonify({"error": "Model could not be loaded."})

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."})

    file = request.files['file']

    try:
        df = pd.read_excel(file)
    except Exception as e:
        return jsonify({"error": f"Error reading Excel file: {e}"})

    required_features = ['Feature1', 'Feature2', 'Feature3']  # Replace with actual feature names
    if not all(col in df.columns for col in required_features):
        return jsonify({"error": f"Excel file must contain columns: {required_features}"})

    features = df[required_features].values
    predictions = model.predict(features)

    df['Predicted Profit'] = predictions
    output_path = "predicted_results.xlsx"
    df.to_excel(output_path, index=False)

    return jsonify({"message": "Prediction completed. Download results.", "output_file": output_path})

@app.route('/predict_single', methods=['POST'])
def predict_single():
    """Predict profit from manual feature inputs."""
    if model is None:
        return jsonify({"error": "Model could not be loaded."})

    data = request.json
    if "features" not in data:
        return jsonify({"error": "Missing 'features' key in request data."})

    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)

    return jsonify({"predicted_profit": float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
