from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

# Define the filename for the exported model and scaler
model_filename = 'logistic_regression_model.joblib'
scaler_filename = 'scaler.joblib'

# Load the trained model
try:
    model = joblib.load(model_filename)
    print(f"Model '{model_filename}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file '{model_filename}' not found. Please ensure the model is exported first.")
    model = None # Handle case where model is not found

# Load the scaler
try:
    scaler = joblib.load(scaler_filename)
    print(f"Scaler '{scaler_filename}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: Scaler file '{scaler_filename}' not found. Please ensure the scaler is exported first.")
    scaler = StandardScaler() # Create a dummy if not found, though this will lead to incorrect predictions if not fitted.
# Initialize Flask app


# Define the filename for the exported model and scaler
model_filename = 'logistic_regression_model.joblib'
scaler_filename = 'scaler.joblib' # Assuming the scaler was also saved, if not, it should be saved first.

# Load the trained model
try:
    model = joblib.load(model_filename)
    print(f"Model '{model_filename}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file '{model_filename}' not found. Please ensure the model is exported first.")
    model = None # Handle case where model is not found



# Ensure the scaler object is available, otherwise create a dummy or raise error
if 'scaler' not in locals():
    print("Warning: Scaler object not found in current session. Model prediction might be inaccurate without proper scaling.")
    # In a production environment, you would load the saved scaler or handle this error appropriately.
    # For now, let's create a placeholder scaler if it's not defined, assuming no scaling, which is incorrect but prevents crash.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler() # This will be an unfitted scaler, leading to incorrect predictions.
    # Ideally, you should save the scaler after fitting and load it here.


# Define the API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Cannot make predictions.'}), 500

    data = request.get_json(force=True)

    if not data:
        return jsonify({'error': 'No input data provided.'}), 400

    # Convert input data to a pandas DataFrame
    # Ensure the order of columns matches the training data
    try:
        input_df = pd.DataFrame([data])
        # Features order must match training data
        expected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        if not all(feature in input_df.columns for feature in expected_features):
            return jsonify({'error': 'Missing one or more required features.'}), 400
        input_df = input_df[expected_features]

        # Preprocess the input data using the loaded scaler
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        # Return the prediction as a JSON response
        return jsonify({
            'prediction': int(prediction[0]),
            'probability_class_0': float(prediction_proba[0][0]),
            'probability_class_1': float(prediction_proba[0][1])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

# For Colab and local testing without blocking the notebook cell, you can use a small server wrapper like:
print("To run the Flask app, execute it in a separate process or use a tool like ngrok.")
print("Example: !pip install flask_ngrok && from flask_ngrok import run_with_ngrok")
print("Then: run_with_ngrok(app) and app.run()")

if__name__=="__main__"
app.run(debug=True)
