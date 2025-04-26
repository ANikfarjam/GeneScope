import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from model import (
    build_catboost_model, train_catboost_model, save_catboost_model, predict_catboost_probs,
    build_mlp_model, get_gene_tuner, get_fusion_tuner
)
from tensorflow.keras.models import load_model
import tensorflow as tf
import keras
import keras_tuner as kt

# Enable unsafe deserialization
keras.config.enable_unsafe_deserialization()

# Enable GPU memory growth
print("Checking available GPUs...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Number of GPUs Available: {len(gpus)}")
else:
    print("No GPU detected. Ensure CUDA is properly installed.")

# === Load pre-split data ===
clinical_train_df = pd.read_csv('./model_data/clinical_train.csv')
clinical_val_df = pd.read_csv('./model_data/clinical_val.csv')
gene_train_df = pd.read_csv('./model_data/gene_exp_train.csv')
gene_val_df = pd.read_csv('./model_data/gene_exp_val.csv')

# Drop Sample columns
clinical_train_df.drop(columns='Samples', errors='ignore', inplace=True)
clinical_val_df.drop(columns='Samples', errors='ignore', inplace=True)
gene_train_df.drop(columns='Samples', errors='ignore', inplace=True)
gene_val_df.drop(columns='Samples', errors='ignore', inplace=True)

# Encode clinical variables
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
clinical_X_train = encoder.fit_transform(clinical_train_df.iloc[:, 1:]).astype(np.float32)
clinical_X_val = encoder.transform(clinical_val_df.iloc[:, 1:]).astype(np.float32)

# Gene expression features
gene_X_train = gene_train_df.iloc[:, 1:].to_numpy(dtype=np.float32)
gene_X_val = gene_val_df.iloc[:, 1:].to_numpy(dtype=np.float32)

# Encode labels
y_train = LabelEncoder().fit_transform(clinical_train_df['Stage'])
y_val = LabelEncoder().fit_transform(clinical_val_df['Stage'])

# === Train CatBoost on clinical ===
cat_model = build_catboost_model()
cat_model = train_catboost_model(cat_model, clinical_X_train, y_train, clinical_X_val, y_val)
save_catboost_model(cat_model, "./best_models1/catboost_clinical")

# === Train MLP using NAS on gene expression ===
gene_tuner = get_gene_tuner(gene_X_train, num_classes=10)
gene_tuner.search(gene_X_train, y_train, validation_data=(gene_X_val, y_val), epochs=10, batch_size=32)
best_gene_model = gene_tuner.get_best_models(1)[0]
best_gene_model.save("./best_models1/best_gene_model.keras")

# === Prepare intermediate fusion inputs ===
clinical_probs_train = predict_catboost_probs(cat_model, clinical_X_train)
clinical_probs_val = predict_catboost_probs(cat_model, clinical_X_val)

gene_probs_train = best_gene_model.predict(gene_X_train)
gene_probs_val = best_gene_model.predict(gene_X_val)

# === Train fusion model using NAS ===
fusion_tuner = get_fusion_tuner(gene_probs_train, clinical_probs_train, num_classes=10)
fusion_tuner.search(
    [gene_probs_train, clinical_probs_train], y_train,
    validation_data=([gene_probs_val, clinical_probs_val], y_val),
    epochs=10, batch_size=32
)
best_fusion_model = fusion_tuner.get_best_models(1)[0]
best_fusion_model.save("./best_models1/best_fusion_model.keras")

print("Training and model saving complete!")
