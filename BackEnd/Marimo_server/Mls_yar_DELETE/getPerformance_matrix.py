import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import load_model
import tensorflow as tf
import warnings

# --- Setup and debug ---
print("📂 Current Working Directory:", os.getcwd())
warnings.filterwarnings("ignore")

# --- Load and clean data ---
df = pd.read_csv("../AHPresults/fina_Stage_unaugmented.csv", low_memory=False)
df.drop_duplicates(subset='Samples', inplace=True)
df.dropna(subset=['Stage'], inplace=True)

drop_cols = [
    'Samples', 'submitter_id', 'barcode', 'Unnamed: 0', 'sample_id',
    'sample', 'sample_submitter_id', 'patient', 'paper_patient',
    'diagnosis_id', 'bcr_patient_barcode', 'pathology_report_uuid',
    'treatments', 'releasable', 'released', 'paper_vital_status',
    'paper_Included_in_previous_marker_papers'
]
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

y = df['Stage']
X = df.drop(columns=['Stage'])

# Filter numeric features and handle NaNs/infs
X = X.select_dtypes(include=[np.number])
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(inplace=True)
y = y.loc[X.index]  # Align labels

# --- Load preprocessing tools and model ---
scaler = joblib.load("scaler.save")
pca = joblib.load("pca.save")
label_encoder = joblib.load("label_encoder.save")

# Load HDF5-based model (despite .keras extension)
model_path = os.path.abspath("vanilla_nn_brca_model.keras")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"🚫 Model file not found at: {model_path}")

nn_model = load_model(model_path, compile=False)
print("✅ Neural network model (HDF5) loaded successfully.")

# Optional: Resave as real .keras format for future compatibility
save_path = "./vanilla_nn_brca_model_v3.keras"
if not os.path.exists(save_path):
    nn_model.save(save_path, save_format="keras_v3")
    print(f"📦 Model also saved in modern `.keras` format to: {save_path}")

# --- Preprocess data ---
X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)
y_encoded = label_encoder.transform(y)
num_classes = len(np.unique(y_encoded))

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# --- KNN Evaluation ---
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

report_knn = classification_report(
    y_test, y_pred_knn, target_names=label_encoder.classes_, output_dict=True)
conf_knn = confusion_matrix(y_test, y_pred_knn)

pd.DataFrame(report_knn).transpose().to_csv("knn_classification_report.csv")
pd.DataFrame(conf_knn, index=label_encoder.classes_, columns=label_encoder.classes_).to_csv("knn_confusion_matrix.csv")

# --- Neural Network Evaluation ---
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
y_pred_nn_probs = nn_model.predict(X_test)
y_pred_nn = np.argmax(y_pred_nn_probs, axis=1)

report_nn = classification_report(
    y_test, y_pred_nn, target_names=label_encoder.classes_, output_dict=True)
conf_nn = confusion_matrix(y_test, y_pred_nn)

pd.DataFrame(report_nn).transpose().to_csv("nn_classification_report.csv")
pd.DataFrame(conf_nn, index=label_encoder.classes_, columns=label_encoder.classes_).to_csv("nn_confusion_matrix.csv")

print("✅ Model evaluation completed and all CSVs saved.")
