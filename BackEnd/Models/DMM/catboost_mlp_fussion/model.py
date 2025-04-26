from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import keras_tuner as kt
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import pandas as pd
from catboost import CatBoostClassifier

# ========== Build MLP Model (used for gene expression) ==========
def build_mlp_model(hp, input_dim, num_classes):
    inputs = Input(shape=(input_dim,), name="gene_input")
    x = inputs

    for i in range(hp.Int("num_layers", 1, 10)):
        x = Dense(
            units=hp.Int(f"units_{i}", min_value=64, max_value=512, step=64),
            activation=hp.Choice(f"activation_{i}", ["relu", "tanh"]),
            name=f"dense_{i}"
        )(x)
        x = Dropout(hp.Float(f"dropout_{i}", 0.0, 0.5, 0.1))(x)

    outputs = Dense(num_classes, activation="softmax", name="gene_output")(x)
    model = Model(inputs, outputs, name="MLP_Model")
    model.compile(
        optimizer=Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ========== CatBoost Clinical Model ==========
def build_catboost_model():
    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=100,
        task_type='CPU'
    )
    return model

# ========== Train CatBoost Model ==========
def train_catboost_model(cat_model, clinical_X_train, y_train, clinical_X_val, y_val):
    cat_model.fit(
        clinical_X_train, y_train,
        eval_set=(clinical_X_val, y_val),
        early_stopping_rounds=50
    )
    return cat_model

# ========== Save CatBoost Model ==========
def save_catboost_model(cat_model, save_path):
    cat_model.save_model(save_path)

# ========== Load CatBoost Model ==========
def load_catboost_model(load_path):
    model = CatBoostClassifier()
    model.load_model(load_path)
    return model

# ========== Predict probabilities from CatBoost ==========
def predict_catboost_probs(cat_model, clinical_X):
    probs = cat_model.predict_proba(clinical_X)
    return probs.astype(np.float32)

# ========== Intermediate-Level Fusion Model (CatBoost + MLP) ==========
def build_catboost_mlp_intermediate_fusion(hp, input_shape_gene=2000, input_shape_clinical=30, num_classes=10):
    gene_input = Input(shape=(input_shape_gene,), name="gene_input")
    clinical_input = Input(shape=(input_shape_clinical,), name="clinical_input")

    concatenated = Concatenate(name="concat_features")([gene_input, clinical_input])

    x = concatenated
    for i in range(hp.Int("fusion_num_layers", 1, 5)):
        x = Dense(
            units=hp.Int(f"fusion_units_{i}", min_value=64, max_value=512, step=64),
            activation=hp.Choice(f"fusion_activation_{i}", ["relu", "tanh"]),
            name=f"fusion_dense_{i}"
        )(x)
        x = Dropout(hp.Float(f"fusion_dropout_{i}", 0.0, 0.5, 0.1))(x)

    outputs = Dense(num_classes, activation='softmax', name="final_output")(x)

    fusion_model = Model(inputs=[gene_input, clinical_input], outputs=outputs, name="CatBoost_MLP_Intermediate_Fusion")
    fusion_model.compile(optimizer=Adam(hp.Float("fusion_lr", 1e-4, 1e-2, sampling="log")),
                         loss="sparse_categorical_crossentropy",
                         metrics=["accuracy"])
    return fusion_model

# ========== Tuner Setup for Gene Expression ==========
def get_gene_tuner(gene_X_train, num_classes=10):
    tuner = kt.Hyperband(
        lambda hp: build_mlp_model(hp, gene_X_train.shape[1], num_classes),
        objective="val_accuracy",
        max_epochs=50,
        factor=3,
        directory="nas_gene_model",
        project_name="breast_cancer_gene"
    )
    return tuner

# ========== Tuner Setup for Fusion Model ==========
def get_fusion_tuner(gene_X_train, clinical_X_train, num_classes=10):
    tuner = kt.Hyperband(
        lambda hp: build_catboost_mlp_intermediate_fusion(hp, input_shape_gene=gene_X_train.shape[1], input_shape_clinical=clinical_X_train.shape[1], num_classes=num_classes),
        objective="val_accuracy",
        max_epochs=50,
        factor=3,
        directory="nas_fusion_model",
        project_name="breast_cancer_fusion"
    )
    return tuner
