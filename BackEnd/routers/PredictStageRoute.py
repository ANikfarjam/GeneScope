from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

predict_bp = Blueprint('predict_bp', __name__)

# Load model and preprocessing tools once
model = tf.keras.models.load_model("../Marimo_server/Mls_yar_TMP/brca_NN_vanila_another.keras", compile=False)
scaler = joblib.load("../Marimo_server/Mls_yar_TMP/scaler.save")
pca = joblib.load("../Marimo_server/Mls_yar_TMP/pca.save")
label_encoder = joblib.load("../Marimo_server/Mls_yar_TMP/label_encoder.save")

# Drop columns function (must match training)
DROP_COLS = [
    'Samples', 'Stage', 'vital_status', 'submitter_id', 'barcode',
    'sample_id', 'sample', 'sample_submitter_id', 'patient', 'paper_patient',
    'diagnosis_id', 'bcr_patient_barcode', 'paper_age_at_initial_pathologic_diagnosis',
    'paper_days_to_birth', 'paper_pathologic_stage', 'ajcc_pathologic_n',
    'ajcc_pathologic_t', 'ajcc_pathologic_m', 'year_of_diagnosis', 'treatments',
    'Unnamed: 0', 'paper_days_to_last_followup', 'days_to_collection',
    'demographic_id', 'initial_weight', 'days_to_birth', 'pathology_report_uuid',
    'age_at_diagnosis', 'age_at_index', 'method_of_diagnosis',
    'sites_of_involvement', 'primary_diagnosis', 'morphology',
    'paper_PARADIGM.Clusters', 'paper_Mutation.Clusters', 'paper_CNV.Clusters',
    'paper_BRCA_Subtype_PAM50', 'paper_miRNA.Clusters', 'paper_DNA.Methylation.Clusters',
    'paper_Included_in_previous_marker_papers', 'paper_mRNA.Clusters',
    'ethnicity', 'preservation_method', 'race', 'laterality',
    'paper_vital_status', 'oct_embedded', 'prior_malignancy',
    'synchronous_malignancy', 'age_is_obfuscated', 'prior_treatment',
    'tissue_or_organ_of_origin', 'icd_10_code'
]

@predict_bp.route('/api/predict-stage', methods=['POST'])
def predict_stage():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        df = pd.read_csv(file)
        df = df.drop(columns=[col for col in DROP_COLS if col in df.columns])
        df = df.select_dtypes(include=[np.number])
        df = df.dropna(axis=1)

        sample_scaled = scaler.transform(df)
        sample_pca = pca.transform(sample_scaled)

        prediction = model.predict(sample_pca)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_stage = label_encoder.classes_[predicted_class]

        return jsonify({
            'predicted_stage': predicted_stage,
            'probabilities': prediction.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
