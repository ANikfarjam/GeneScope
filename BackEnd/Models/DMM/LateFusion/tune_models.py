import os
import json
import gc
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from rich.progress import Progress
from keras.models import clone_model

from model import get_tuners, build_score_fusion_model

# Enable GPU memory growth
print("Checking available GPUs...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Number of GPUs Available: {len(gpus)}")
else:
    print("No GPU detected. Ensure CUDA is properly installed.")

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

# Load or build datasets
if config["stats"] == False:
    raise RuntimeError("You need to run preprocessing first to generate stats.")
else:
    clinical_train_df = pd.read_csv('./model_data/clinical_train.csv')
    clinical_val_df = pd.read_csv('./model_data/clinical_val.csv')
    clinical_train_df.drop(columns='Samples', inplace=True)
    clinical_val_df.drop(columns='Samples', inplace=True)

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    clinical_train_df.iloc[:, 1:] = encoder.fit_transform(clinical_train_df.iloc[:, 1:]).astype(np.float32)
    clinical_val_df.iloc[:, 1:] = encoder.transform(clinical_val_df.iloc[:, 1:]).astype(np.float32)

    clinical_X_train = clinical_train_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    clinical_X_val = clinical_val_df.iloc[:, 1:].to_numpy(dtype=np.float32)

    gene_exp_train_df = pd.read_csv('./model_data/gene_exp_train.csv')
    gene_exp_val_df = pd.read_csv('./model_data/gene_exp_val.csv')

with Progress() as p:
    task = p.add_task('Setting up and tuning models', total=5)

    # Encode target labels
    clinical_y_train = clinical_train_df['Stage']
    clinical_y_val = clinical_val_df['Stage']
    stage_encoder = LabelEncoder()
    clinical_y_train = stage_encoder.fit_transform(clinical_y_train)
    clinical_y_val = stage_encoder.transform(clinical_y_val)

    gene_y_train = stage_encoder.transform(gene_exp_train_df['Stage'])
    gene_y_val = stage_encoder.transform(gene_exp_val_df['Stage'])

    gene_X_train = gene_exp_train_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    gene_X_val = gene_exp_val_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    print("Gene input shape:", gene_X_train.shape)
    print("Clinical input shape:", clinical_X_train.shape)

    # Run NAS
    gene_tuner, clinical_tuner = get_tuners(
        clinical_X_train=clinical_X_train,
        gene_X_train=gene_X_train,
        num_classes=10
    )

    print("Running clinical tuner...")
    clinical_tuner.search(clinical_X_train, clinical_y_train,
                          validation_data=(clinical_X_val, clinical_y_val),
                          epochs=10, batch_size=32)
    p.advance(task)
    best_clinical_model = clinical_tuner.get_best_models(1)[0]
    best_clinical_model.save('./best_models1/best_clinical_model.keras')

    del clinical_tuner
    tf.keras.backend.clear_session()
    gc.collect()

    print("Running gene tuner...")
    gene_tuner.search(gene_X_train, gene_y_train,
                      validation_data=(gene_X_val, gene_y_val),
                      epochs=10, batch_size=32)
    p.advance(task)
    best_gene_model = gene_tuner.get_best_models(1)[0]
    best_gene_model.save('./best_models1/best_gene_model.keras')

    del gene_tuner
    tf.keras.backend.clear_session()
    gc.collect()

    # Reload best models
    best_clinical_model = tf.keras.models.load_model('./best_models1/best_clinical_model.keras')
    best_gene_model = tf.keras.models.load_model('./best_models1/best_gene_model.keras')
    p.advance(task)

    # Fix layer naming conflicts
    def rebuild_model_with_prefix(saved_model, prefix):
        config = saved_model.get_config()
        config['name'] = f"{prefix}_model"
        for layer in config['layers']:
            layer['config']['name'] = f"{prefix}_{layer['config']['name']}"
        rebuilt_model = tf.keras.Model.from_config(config)
        rebuilt_model.set_weights(saved_model.get_weights())
        return rebuilt_model

    # After loading best models:
    best_clinical_model = tf.keras.models.load_model('./best_models1/best_clinical_model.keras')
    best_gene_model = tf.keras.models.load_model('./best_models1/best_gene_model.keras')

    # Clean rename & rebuild
    best_clinical_model = rebuild_model_with_prefix(best_clinical_model, "clinical")
    best_gene_model = rebuild_model_with_prefix(best_gene_model, "gene")

    print("Building score-level fusion model...")
    fusion_model = build_score_fusion_model(
        gene_model=best_gene_model,
        clinical_model=best_clinical_model,
        alpha=0.4,  
        beta=0.6,
        input_shape_gene=gene_X_train.shape[1],
        input_shape_clinical=clinical_X_train.shape[1],
        num_classes=10
    )
    p.advance(task)

    print("Saving fusion model...")
    fusion_model.save('./best_models1/fusion_model.keras')
    p.advance(task)

    print("Tuning and fusion complete.")
