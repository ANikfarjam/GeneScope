from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

predict_bp = Blueprint('predict_bp', __name__)

# Load model and pipeline
model = tf.keras.models.load_model("C:/Users/pc/Desktop/projects/GeneScope/BackEnd/Models/MLP/brca_model_with_clinical.keras")
pipeline = joblib.load("C:/Users/pc/Desktop/projects/GeneScope/BackEnd/Models/MLP/pipeline_with_pca.save")
label_encoder = joblib.load("C:/Users/pc/Desktop/projects/GeneScope/BackEnd/Models/MLP/label_encoder.save")
expected_columns = joblib.load("C:/Users/pc/Desktop/projects/GeneScope/BackEnd/Models/MLP/expected_columns.save")

DROP_COLS = [
    'site_of_resection_or_biopsy', 'tumor_descriptor', 'sample_type_id', 'definition', 'primary_site',
    'name', 'disease_type', 'shortLetterCode', 'sample_type', 'project_id', 'classification_of_tumor',
    'specimen_type', 'state', 'is_ffpe', 'tissue_type', 'composition', 'paper_Tumor.Type', 'gender',
    'days_to_diagnosis', 'releasable', 'diagnosis_is_primary_disease', 'released'
]

@predict_bp.route('/api/predict-stage', methods=['POST'])
def predict_stage():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        file = request.files['file']
        df = pd.read_csv(file)

        # Drop irrelevant columns
        df = df.drop(columns=[col for col in DROP_COLS if col in df.columns])

        # Ensure only expected columns are used
        df = df[[col for col in expected_columns if col in df.columns]]

        # Replace infs and drop missing
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        if df.empty:
            return jsonify({'error': 'Input data is empty after cleaning'}), 400

        # Apply full preprocessing pipeline (scaling, encoding, PCA)
        X_transformed = pipeline.transform(df)

        # Predict
        prediction = model.predict(X_transformed)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_stage = label_encoder.classes_[predicted_class]

        return jsonify({
            'predicted_stage': predicted_stage,
            'probabilities': prediction.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
