from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
import keras_tuner as kt

def build_mlp_model(hp, input_shape, num_classes):
    model = Sequential()
    model.add(keras.Input(shape=(input_shape,)))

    # Tune number of hidden layers: 1–4
    for i in range(hp.Int("num_layers", 1, 4)):
        model.add(Dense(
            units=hp.Int(f"units_{i}", min_value=64, max_value=1024, step=64),
            activation=hp.Choice(f"activation_{i}", ["relu", "tanh"]),
        ))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float(f"dropout_{i}", min_value=0.0, max_value=0.5, step=0.1)))

    # Output layer (for multiclass)
    model.add(Dense(num_classes, activation="softmax"))

    # Compile with tunable learning rate
    model.compile(
        optimizer=Adam(hp.Float("lr", 1e-5, 1e-2, sampling="log")),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
