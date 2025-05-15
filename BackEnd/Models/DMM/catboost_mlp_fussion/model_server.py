from flask import Flask, request, jsonify
from model import load_catboost_model, predict_catboost_probs
from tensorflow.keras.models import load_model
import numpy as np
import json
import os
from sklearn.preprocessing import OrdinalEncoder
import tensorflow as tf


app = Flask(__name__)

# === Load models once at startup ===
cat_model = load_catboost_model("./best_models1/catboost_clinical")
gene_model = load_model("./best_models1/best_gene_model.keras")
fusion_model = load_model("./best_models1/best_fusion_model.keras")

# === Load encoder ===
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoder.fit(np.load("./best_models1/clinical_encoder_fit.npy", allow_pickle=True))

@app.route("/predict_stage", methods=["POST"])
def predict_stage():
    try:
        data = request.get_json()
        full_input = np.array(data["features"], dtype=np.float32)

        if full_input.shape[0] != 2030:
            return jsonify({"error": "Expected 2030 features (30 clinical + 2000 gene)"}), 400

        # === Split input ===
        clinical_raw = full_input[:30].reshape(1, -1)
        gene_raw = full_input[30:].reshape(1, -1)

        # === Encode + Predict ===
        clinical_encoded = encoder.transform(clinical_raw)
        clinical_probs = predict_catboost_probs(cat_model, clinical_encoded)
        gene_probs = gene_model.predict(gene_raw)
        fusion_probs = fusion_model.predict([gene_probs, clinical_probs])[0]

        # === Format Output ===
        stage_output = {f"Stage {i}": float(round(prob, 4)) for i, prob in enumerate(fusion_probs)}
        return jsonify({"stage_probabilities": stage_output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
