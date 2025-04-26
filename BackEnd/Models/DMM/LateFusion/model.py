from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
import tensorflow as tf


# ========== Build MLP Model (used independently for gene & clinical) ==========
def build_mlp_model(hp, input_dim, num_classes):
    inputs = Input(shape=(input_dim,))
    x = inputs

    for i in range(hp.Int("num_layers", 1, 5)):
        x = Dense(
            units=hp.Int(f"units_{i}", min_value=64, max_value=512, step=64),
            activation=hp.Choice(f"activation_{i}", ["relu", "tanh"]),
            name=f"dense_{i}"
        )(x)
        x = BatchNormalization(name=f"bn_{i}")(x)
        x = Dropout(hp.Float(f"dropout_{i}", min_value=0.0, max_value=0.5, step=0.1), name=f"dropout_{i}")(x)

    outputs = Dense(num_classes, activation="softmax", name="output")(x)
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ========== Score-Level Fusion Model ==========
def build_score_fusion_model(gene_model, clinical_model, alpha=0.5, beta=0.5,
                              input_shape_gene=400, input_shape_clinical=25, num_classes=10):
    assert abs(alpha + beta - 1.0) < 1e-6, "Alpha and Beta must sum to 1"

    input_gene = Input(shape=(input_shape_gene,), name="gene_input")
    input_clinical = Input(shape=(input_shape_clinical,), name="clinical_input")

    # Get softmax output from both pretrained models
    gene_out = gene_model(input_gene)
    clinical_out = clinical_model(input_clinical)

    # Weighted score-level fusion
    fused = tf.keras.layers.Add(name="fusion_score")([
        tf.keras.layers.Lambda(lambda x: x * beta, name="scale_gene")(gene_out),
        tf.keras.layers.Lambda(lambda x: x * alpha, name="scale_clinical")(clinical_out)
    ])

    fusion_model = Model(inputs=[input_gene, input_clinical], outputs=fused, name="fusion_model")
    fusion_model.compile(optimizer=Adam(1e-4),
                         loss="sparse_categorical_crossentropy",
                         metrics=["accuracy"])
    return fusion_model


# ========== Tuner Setup ==========
def get_tuners(clinical_X_train, gene_X_train, num_classes=10):
    input_shape_gene = gene_X_train.shape[1]
    input_shape_clinical = clinical_X_train.shape[1]

    tuner_gene = kt.Hyperband(
        lambda hp: build_mlp_model(hp, input_shape_gene, num_classes),
        objective="val_accuracy",
        max_epochs=50,
        factor=3,
        directory="nas_gene_model",
        project_name="breast_cancer_gene"
    )

    tuner_clinical = kt.Hyperband(
        lambda hp: build_mlp_model(hp, input_shape_clinical, num_classes),
        objective="val_accuracy",
        max_epochs=50,
        factor=3,
        directory="nas_clinical_model",
        project_name="breast_cancer_clinical"
    )

    return tuner_gene, tuner_clinical
