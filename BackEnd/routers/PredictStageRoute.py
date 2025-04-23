from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

predict_bp = Blueprint('predict_bp', __name__)

# Load model and preprocessing tools
model = tf.keras.models.load_model("C:/Users/pc/Desktop/projects/GeneScope/BackEnd/Models/MLP/vanilla_nn_brca_model.keras")
scaler = joblib.load("C:/Users/pc/Desktop/projects/GeneScope/BackEnd/Models/MLP/scaler.save")
pca = joblib.load("C:/Users/pc/Desktop/projects/GeneScope/BackEnd/Models/MLP/pca.save")
label_encoder = joblib.load("C:/Users/pc/Desktop/projects/GeneScope/BackEnd/Models/MLP/label_encoder.save")

@predict_bp.route('/api/predict-stage', methods=['POST'])
def predict_stage():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        file = request.files['file']
        df = pd.read_csv(file)

        # Drop non-numeric columns
        df = df.select_dtypes(include=[np.number])

        # Replace inf and drop rows with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        if df.empty:
            return jsonify({'error': 'Input data is empty after cleaning'}), 400

        # Scale → PCA
        X_scaled = scaler.transform(df)
        X_pca = pca.transform(X_scaled)

        # Predict
        prediction = model.predict(X_pca)
        pred_index = np.argmax(prediction, axis=1)[0]
        pred_stage = label_encoder.classes_[pred_index]

        return jsonify({
            'predicted_stage': pred_stage,
            'probabilities': prediction[0].tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
