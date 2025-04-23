from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

# ========== Build MLP Model (Tunable) ==========
def build_mlp_model(hp, input_shape, num_classes):
    inputs = Input(shape=(input_shape,))
    x = inputs

    for i in range(hp.Int("num_layers", 1, 6)):
        x = Dense(
            units=hp.Int(f"units_{i}", min_value=64, max_value=1024, step=64),
            activation=hp.Choice(f"activation_{i}", ["relu", "tanh"]),
            name=f"dense_{i}"
        )(x)
        x = BatchNormalization(name=f"bn_{i}")(x)
        x = Dropout(hp.Float(f"dropout_{i}", min_value=0.0, max_value=0.5, step=0.1), name=f"dropout_{i}")(x)

    outputs = Dense(num_classes, activation="softmax", name="output")(x)
    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(hp.Float("lr", 1e-5, 1e-2, sampling="log")),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ========== Tuner Setup Function ==========
def get_tuners(input_shape_gene=500, input_shape_clinical=25, num_classes=10):
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

# ========== Late Fusion Model Builder ==========
def build_late_fusion_model(best_gene_model, best_clinical_model, input_shape_gene=400, input_shape_clinical=25, num_classes=10):
    input_gene = Input(shape=(input_shape_gene,), name="gene_input")
    input_clinical = Input(shape=(input_shape_clinical,), name="clinical_input")

    gene_features = best_gene_model.get_layer(index=-2)(input_gene)
    clinical_features = best_clinical_model.get_layer(index=-2)(input_clinical)

    fused = Concatenate()([gene_features, clinical_features])
    fused = Dense(128, activation="relu")(fused)
    fused = Dropout(0.3)(fused)
    output = Dense(num_classes, activation="softmax")(fused)

    model = Model(inputs=[input_gene, input_clinical], outputs=output)
    model.compile(optimizer=Adam(1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
