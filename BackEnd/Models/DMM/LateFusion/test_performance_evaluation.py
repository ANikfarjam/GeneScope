import numpy as np
import pandas as pd
import os
import json
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from tensorflow.keras.models import load_model
import keras

# === Load and preprocess test data ===
clinical_test = pd.read_csv('./model_data/clinical_test.csv')
gene_test = pd.read_csv('./model_data/gene_exp_test.csv')

# --- Drop Sample columns if exists ---
if 'Samples' in clinical_test.columns:
    clinical_test.drop(columns='Samples', inplace=True)
if 'Samples' in gene_test.columns:
    gene_test.drop(columns='Samples', inplace=True)

# === Ordinal encode clinical features ===
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
clinical_features = encoder.fit_transform(clinical_test.iloc[:, 1:]).astype(np.float32)

# === Label encode target (Stage) ===
stage_encoder = LabelEncoder()
y_true = stage_encoder.fit_transform(clinical_test['Stage'])

# === Prepare gene expression features ===
gene_features = gene_test.iloc[:, 1:].to_numpy(dtype=np.float32)

# === Load trained fusion model (with unsafe deserialization) ===
keras.config.enable_unsafe_deserialization()
fusion_model = load_model('./best_models1/fusion_model.keras')

# === Predict and decode ===
y_prob = fusion_model.predict([gene_features, clinical_features])
y_pred = np.argmax(y_prob, axis=1)

# === Metrics on Test Set ===
acc_test = accuracy_score(y_true, y_pred)
pre_test = precision_score(y_true, y_pred, average='macro')
sn_test = recall_score(y_true, y_pred, average='macro')
mcc_test = matthews_corrcoef(y_true, y_pred)

# === Specificity ===
def compute_specificity(y_true, y_pred, n_classes):
    sp_list = []
    for i in range(n_classes):
        binary_true = (y_true == i)
        binary_pred = (y_pred == i)
        tn = np.sum((~binary_true) & (~binary_pred))
        fp = np.sum((~binary_true) & binary_pred)
        sp = tn / (tn + fp + 1e-6)
        sp_list.append(sp)
    return np.mean(sp_list)

sp_test = compute_specificity(y_true, y_pred, n_classes=y_prob.shape[1])

# === ROC-AUC per class ===
auc_scores = []
for i in range(y_prob.shape[1]):
    y_binary = (y_true == i).astype(int)
    auc = roc_auc_score(y_binary, y_prob[:, i])
    auc_scores.append((f'Class {i} ({stage_encoder.classes_[i]})', auc))

# === Load NAS tuner validation scores ===
def extract_best_val_accuracy(tuner_dir):
    try:
        with open(os.path.join(tuner_dir, 'trial_summary.json'), 'r') as f:
            trials = json.load(f)['trials']
        best_val_acc = 0
        for trial in trials:
            trial_val_acc = trial['score']
            if trial_val_acc > best_val_acc:
                best_val_acc = trial_val_acc
        return best_val_acc
    except Exception as e:
        print(f"Warning: could not load NAS tuner results from {tuner_dir}: {e}")
        return None

clinical_val_acc = extract_best_val_accuracy('./nas_clinical_model/breast_cancer_clinical')
gene_val_acc = extract_best_val_accuracy('./nas_gene_model/breast_cancer_gene')

# === Save results ===
metrics_summary = pd.DataFrame({
    "Metric": [
        "Test Accuracy", "Test Precision", "Test Sensitivity", "Test Specificity", "Test MCC",
        "Clinical NAS Best Validation Accuracy",
        "Gene Expression NAS Best Validation Accuracy"
    ],
    "Score": [
        acc_test, pre_test, sn_test, sp_test, mcc_test,
        clinical_val_acc, gene_val_acc
    ]
})

roc_auc_df = pd.DataFrame(auc_scores, columns=["Metric", "Score"])
results = pd.concat([metrics_summary, roc_auc_df])

results.to_csv("fusion_model_evaluation.csv", index=False)
print("\nEvaluation complete. Results saved to 'fusion_model_evaluation.csv'")
