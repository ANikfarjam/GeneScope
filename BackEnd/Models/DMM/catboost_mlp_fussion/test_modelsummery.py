import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, accuracy_score, classification_report
from model import load_catboost_model, predict_catboost_probs
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import os
import tensorflow as tf
# Enable GPU memory growth
print("Checking available GPUs...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Number of GPUs Available: {len(gpus)}")
else:
    print("No GPU detected. Ensure CUDA is properly installed.")
# === Load validation data ===
clinical_val_df = pd.read_csv('./model_data/clinical_val.csv')
gene_val_df = pd.read_csv('./model_data/gene_exp_val.csv')

clinical_val_df.drop(columns='Samples', errors='ignore', inplace=True)
gene_val_df.drop(columns='Samples', errors='ignore', inplace=True)

# Load test data
clinical_test_df = pd.read_csv('./model_data/clinical_test.csv')
gene_test_df = pd.read_csv('./model_data/gene_exp_test.csv')

clinical_test_df.drop(columns='Samples', errors='ignore', inplace=True)
gene_test_df.drop(columns='Samples', errors='ignore', inplace=True)

# Encode clinical variables
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
clinical_X_val = encoder.fit_transform(clinical_val_df.iloc[:, 1:]).astype(np.float32)
clinical_X_test = encoder.transform(clinical_test_df.iloc[:, 1:]).astype(np.float32)

# Gene expression features
gene_X_val = gene_val_df.iloc[:, 1:].to_numpy(dtype=np.float32)
gene_X_test = gene_test_df.iloc[:, 1:].to_numpy(dtype=np.float32)

# Encode labels
y_val = LabelEncoder().fit_transform(clinical_val_df['Stage'])
y_test = LabelEncoder().fit_transform(clinical_test_df['Stage'])

# === Load best models ===
cat_model = load_catboost_model("./best_models1/catboost_clinical")
best_gene_model = load_model("./best_models1/best_gene_model.keras")
best_fusion_model = load_model("./best_models1/best_fusion_model.keras")

# === Show MLP and Fusion Model summaries ===
print("\n===== Best Gene MLP Model Summary =====")
best_gene_model.summary()

print("\n===== Best Fusion Model Summary =====")
best_fusion_model.summary()

# === Predict on validation set ===
clinical_probs_val = predict_catboost_probs(cat_model, clinical_X_val)
gene_probs_val = best_gene_model.predict(gene_X_val)

fusion_preds_probs_val = best_fusion_model.predict([gene_probs_val, clinical_probs_val])
fusion_preds_val = np.argmax(fusion_preds_probs_val, axis=1)

# === Predict on test set ===
clinical_probs_test = predict_catboost_probs(cat_model, clinical_X_test)
gene_probs_test = best_gene_model.predict(gene_X_test)

fusion_preds_probs_test = best_fusion_model.predict([gene_probs_test, clinical_probs_test])
fusion_preds_test = np.argmax(fusion_preds_probs_test, axis=1)

# === Define performance calculation function ===
def calculate_metrics(y_true, y_pred, y_probs):
    auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    cm = confusion_matrix(y_true, y_pred)
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (FP + FN + TP)

    epsilon = 1e-7
    Sn = (TP / (TP + FN + epsilon)).mean()
    Sp = (TN / (TN + FP + epsilon)).mean()
    Acc = accuracy_score(y_true, y_pred)
    Pre = precision_score(y_true, y_pred, average='macro', zero_division=0)
    MCC_numerator = (TP.sum() * TN.sum()) - (FP.sum() * FN.sum())
    MCC_denominator = np.sqrt((TP.sum() + FP.sum()) * (TP.sum() + FN.sum()) * (TN.sum() + FP.sum()) * (TN.sum() + FN.sum()))
    MCC = MCC_numerator / (MCC_denominator + epsilon)
    return auc, Sn, Sp, Acc, Pre, MCC

# === Calculate for validation and test ===
val_auc, val_Sn, val_Sp, val_Acc, val_Pre, val_MCC = calculate_metrics(y_val, fusion_preds_val, fusion_preds_probs_val)
test_auc, test_Sn, test_Sp, test_Acc, test_Pre, test_MCC = calculate_metrics(y_test, fusion_preds_test, fusion_preds_probs_test)

# === Save overall metrics ===
overall_results = pd.DataFrame({
    "Metric": ["AUC", "Sensitivity", "Specificity", "Accuracy", "Precision", "MCC"],
    "Validation": [val_auc, val_Sn, val_Sp, val_Acc, val_Pre, val_MCC],
    "Test": [test_auc, test_Sn, test_Sp, test_Acc, test_Pre, test_MCC]
})

# === Save per-class AUCs ===
val_class_auc = roc_auc_score(y_val, fusion_preds_probs_val, multi_class='ovr', average=None)
test_class_auc = roc_auc_score(y_test, fusion_preds_probs_test, multi_class='ovr', average=None)

class_results = pd.DataFrame({
    "Class": [f"Class {i}" for i in range(len(val_class_auc))],
    "Validation AUC": val_class_auc,
    "Test AUC": test_class_auc
})

# Create results directory if it doesn't exist
os.makedirs("./results", exist_ok=True)

overall_results.to_csv("./results/overall_performance_val_test.csv", index=False)
class_results.to_csv("./results/class_auc_val_test.csv", index=False)

# === Print ===
print("\n===== Overall Performance (Validation vs Test) =====")
print(overall_results)

print("\n===== Per-Class AUCs (Validation vs Test) =====")
print(class_results)
