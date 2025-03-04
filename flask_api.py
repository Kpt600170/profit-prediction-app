from flask import Flask, request, jsonify, send_file
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the trained ElasticNet model
MODEL_PATH = "C:/Users/Keshavi/Downloads/New folder (2)/elastic_net_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


@app.route('/')
def home():
    """Home route for API."""
    return "Welcome to the ElasticNet Profit Prediction API! 🚀", 200


@app.route('/predict_excel', methods=['POST'])
def predict_from_excel():
    """Predict profits from an uploaded Excel or CSV file."""
    if model is None:
        return jsonify({"error": "Model could not be loaded."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files['file']

    # Detect file type and read accordingly
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file, engine="openpyxl")  # Fix for excel engine issue
        else:
            return jsonify({"error": "Unsupported file format. Please upload an Excel (.xls, .xlsx) or CSV file."}), 400
    except Exception as e:
        return jsonify({"error": f"Error reading file: {e}"}), 400

    # Ensure required columns are present
    required_features = ['R&D Spend', 'Administration ', 'Market Spend']  # Replace with actual feature names
    if not all(col in df.columns for col in required_features):
        return jsonify({"error": f"File must contain columns: {required_features}"}), 400

    # Make predictions
    features = df[required_features].values
    predictions = model.predict(features)

    # Save predictions
    df['Predicted Profit'] = predictions
    output_path = "predicted_results.xlsx"
    df.to_excel(output_path, index=False, engine="openpyxl")

    return send_file(output_path, as_attachment=True)


@app.route('/predict_single', methods=['POST'])
def predict_single():
    """Predict profit from manual feature inputs."""
    if model is None:
        return jsonify({"error": "Model could not be loaded."}), 500

    data = request.json
    if "features" not in data:
        return jsonify({"error": "Missing 'features' key in request data."}), 400

    try:
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"predicted_profit": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 400


if __name__ == '__main__':
    app.run(debug=True)
