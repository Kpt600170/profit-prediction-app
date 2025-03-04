from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the trained ElasticNet model
MODEL_PATH = "C:/Users/Keshavi/Downloads/New folder (2)/elastic_net_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# API endpoint to handle file uploads and predictions
@app.route('/predict_csv', methods=['POST'])
def predict_from_csv():
    if model is None:
        return jsonify({"error": "Model could not be loaded."})

    # Check if a file is uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."})

    file = request.files['file']

    # Read the CSV file
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Error reading CSV file: {e}"})

    # Ensure required feature columns exist
    required_features = ['R&D Spend', 'Administration', 'Marketing Spend']  # Replace with actual feature names
    if not all(col in df.columns for col in required_features):
        return jsonify({"error": f"CSV file must contain columns: {required_features}"})

    # Extract features
    features = df[required_features].values

    # Make predictions
    predictions = model.predict(features)

    # Add predictions to DataFrame
    df['Predicted Profit'] = predictions

    # Save results to a new CSV file
    output_path = "predicted_results.csv"
    df.to_csv(output_path, index=False)

    return jsonify({"message": "Prediction completed. Download results.", "output_file": output_path})

if __name__ == '__main__':
    app.run(debug=True)
