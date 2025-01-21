from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os

app = Flask(__name__)

# Load the model and preprocessor from the saved pickle file
with open('model/downtime_model.pkl', 'rb') as file:
    model_data = pickle.load(file)
    model = model_data['model']
    preprocessor = model_data['preprocessor']

# Upload Endpoint
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith('.csv'):
        file_path = os.path.join('data', 'uploaded_data.csv')
        file.save(file_path)
        return jsonify({"message": "File uploaded successfully!"}), 200
    return jsonify({"error": "Invalid file type"}), 400

# Train Endpoint
@app.route('/train', methods=['POST'])
def train_model():
    try:
        data = pd.read_csv('data/uploaded_data.csv')
        data = data.drop(columns=["UDI", "Product ID"])
        
        X = data.drop(columns=["Target", "Failure Type"])
        y = data["Target"]
        
        categorical_cols = ["Type"]
        numerical_cols = [col for col in X.columns if col not in categorical_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_cols),
                ("cat", OneHotEncoder(drop="first", handle_unknown='ignore'), categorical_cols),
            ]
        )

        preprocessor.fit(X)
        X_preprocessed = preprocessor.transform(X)

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

        model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
        model.fit(X_train, y_train)

        # Evaluate performance
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        with open('model/downtime_model.pkl', 'wb') as file:
            pickle.dump({"model": model, "preprocessor": preprocessor}, file)

        return jsonify({
            "message": "Model trained successfully!",
            "accuracy": accuracy,
            "f1_score": f1
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Predict Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        example_df = pd.DataFrame([data])

        example_preprocessed = preprocessor.transform(example_df)

        example_prediction = model.predict(example_preprocessed)

        prediction_label = "Yes" if example_prediction[0] == 1 else "No"
        confidence = model.predict_proba(example_preprocessed)[0][example_prediction[0]]
        
        return jsonify({
            "Downtime": prediction_label,
            "Confidence": confidence
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
