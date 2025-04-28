from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
predict_bp = Blueprint('predict_bp', __name__)

# Load model and preprocessing tools
MLP_DIR = os.path.join("Models", "MLP")

model = tf.keras.models.load_model(os.path.join(MLP_DIR, "vanilla_nn_brca_model.keras"))
scaler = joblib.load(os.path.join(MLP_DIR, "scaler.save"))
pca = joblib.load(os.path.join(MLP_DIR, "pca.save"))
label_encoder = joblib.load(os.path.join(MLP_DIR, "label_encoder.save"))
@predict_bp.route('/api/predict-stage', methods=['POST'])
def predict_stage():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        file = request.files['file']
        df = pd.read_csv(file)

        clinical_info = df[[
            'year_of_diagnosis',
            'age_at_index', 'initial_weight'
        ]].iloc[0].to_dict()

        df = df.select_dtypes(include=[np.number])
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        if df.empty:
            return jsonify({'error': 'Input data is empty after cleaning'}), 400

        X_scaled = scaler.transform(df)
        X_pca = pca.transform(X_scaled)

        prediction = model.predict(X_pca)
        pred_index = np.argmax(prediction, axis=1)[0]
        pred_stage = label_encoder.classes_[pred_index]

        return jsonify({
            'clinical_info': clinical_info,
            'predicted_stage': pred_stage,
            'probabilities': prediction[0].tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
