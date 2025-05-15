import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.utils import model_to_dot
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import keras_tuner as kt
import os

# === GPU Setup ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Using GPU: {gpus[0].name}")
else:
    print("No GPU detected.")

# === Load and clean dataset ===
data = pd.read_csv('../../LateFusion/fina_Stage_unaugmented.csv', low_memory=False).dropna()

gene_data = data.iloc[:, -2000:].astype(np.float32)
clinical_raw = data.iloc[:, 1:-2000]

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
clinical_encoded = encoder.fit_transform(clinical_raw)
clinical_data = clinical_encoded.astype(np.float32)

X = np.hstack([clinical_data, gene_data])
input_dim = X.shape[1]

label_encoder = LabelEncoder()
stage_labels = label_encoder.fit_transform(data["Stage"])
num_classes = len(label_encoder.classes_)
stage_one_hot = tf.keras.utils.to_categorical(stage_labels, num_classes)
latent_dim = 100

class GeneratorHyperModel(kt.HyperModel):
    def build(self, hp):
        stage_input = layers.Input(shape=(num_classes,))
        noise_input = layers.Input(shape=(latent_dim,))
        x = layers.Concatenate()([stage_input, noise_input])
        for i in range(hp.Int("gen_layers", 1, 3)):
            x = layers.Dense(hp.Int(f"gen_units_{i}", 256, 1024, step=256), activation='relu')(x)
        out = layers.Dense(input_dim, activation='tanh')(x)
        model = Model([stage_input, noise_input], out, name="Generator")
        model.compile(optimizer='adam', loss='mse')
        return model

class DiscriminatorHyperModel(kt.HyperModel):
    def build(self, hp):
        stage_input = layers.Input(shape=(num_classes,))
        data_input = layers.Input(shape=(input_dim,))
        x = layers.Concatenate()([stage_input, data_input])
        for i in range(hp.Int("disc_layers", 1, 3)):
            x = layers.Dense(hp.Int(f"disc_units_{i}", 256, 1024, step=256), activation='relu')(x)
        out = layers.Dense(1, activation='sigmoid')(x)
        model = Model([stage_input, data_input], out, name="Discriminator")
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

print("Tuning generator...")
gen_tuner = kt.Hyperband(
    GeneratorHyperModel(),
    objective="val_loss",
    max_epochs=10,
    directory="kt_gen_tuner",
    project_name="cgan_generator"
)
gen_dummy_y = X[:100]
gen_dummy_stage = stage_one_hot[:100]
gen_dummy_noise = tf.random.normal((100, latent_dim))
gen_tuner.search([gen_dummy_stage, gen_dummy_noise], gen_dummy_y, validation_split=0.2, epochs=5, verbose=0)
best_generator = gen_tuner.get_best_models(1)[0]

print("Tuning discriminator...")
disc_tuner = kt.Hyperband(
    DiscriminatorHyperModel(),
    objective="val_loss",
    max_epochs=10,
    directory="kt_disc_tuner",
    project_name="cgan_discriminator"
)
disc_dummy_fake = best_generator.predict([gen_dummy_stage, gen_dummy_noise])
disc_tuner.search([gen_dummy_stage, disc_dummy_fake], np.zeros((100, 1)), validation_split=0.2, epochs=5, verbose=0)
best_discriminator = disc_tuner.get_best_models(1)[0]

class ConditionalGAN(Model):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.g_loss = tf.keras.metrics.Mean(name="g_loss")
        self.d_loss = tf.keras.metrics.Mean(name="d_loss")

    def compile(self, g_opt, d_opt):
        super().compile()
        self.g_opt = g_opt
        self.d_opt = d_opt

    def train_step(self, data):
        stage_labels, real_data = data[0]
        batch_size = tf.shape(real_data)[0]
        noise = tf.random.normal((batch_size, latent_dim))

        fake_data = self.generator([stage_labels, noise])
        with tf.GradientTape() as d_tape:
            real_output = self.discriminator([stage_labels, real_data])
            fake_output = self.discriminator([stage_labels, fake_data])
            d_loss = 0.5 * (self.loss_fn(tf.ones_like(real_output), real_output) + self.loss_fn(tf.zeros_like(fake_output), fake_output))
        grads_d = d_tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_opt.apply_gradients(zip(grads_d, self.discriminator.trainable_weights))

        noise = tf.random.normal((batch_size, latent_dim))
        with tf.GradientTape() as g_tape:
            fake_data = self.generator([stage_labels, noise])
            fake_output = self.discriminator([stage_labels, fake_data])
            g_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
        grads_g = g_tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_opt.apply_gradients(zip(grads_g, self.generator.trainable_weights))

        self.g_loss.update_state(g_loss)
        self.d_loss.update_state(d_loss)
        return {"g_loss": self.g_loss.result(), "d_loss": self.d_loss.result()}

cgan = ConditionalGAN(best_generator, best_discriminator)
cgan.compile(
    g_opt=tf.keras.optimizers.Adam(1e-4),
    d_opt=tf.keras.optimizers.Adam(1e-4)
)
cgan.fit(x=[stage_one_hot, X], epochs=100, batch_size=64)

os.makedirs("model_plots", exist_ok=True)
flattened_input_stage = Input(shape=(num_classes,), name="stage_input")
flattened_input_noise = Input(shape=(latent_dim,), name="noise_input")
flattened_gen_out = best_generator([flattened_input_stage, flattened_input_noise])
flattened_disc_out = best_discriminator([flattened_input_stage, flattened_gen_out])
flattened_cgan = Model([flattened_input_stage, flattened_input_noise], flattened_disc_out, name="Flattened_cGAN")
flattened_dot = model_to_dot(flattened_cgan, show_shapes=True, dpi=100)
with open("model_plots/flattened_cgan.dot", "w") as f:
    f.write(flattened_dot.to_string())
flattened_dot.write_svg("model_plots/flattened_cgan.svg")
flattened_dot.write_png("model_plots/flattened_cgan.png")

print("✅ Flattened model graph saved as PNG and SVG.")

os.makedirs("synthetic_by_stage", exist_ok=True)
clinical_cols = list(clinical_raw.columns)
gene_cols = list(gene_data.columns)
all_cols = clinical_cols + gene_cols

for i in range(num_classes):
    label = np.zeros((10, num_classes))
    label[:, i] = 1
    noise = tf.random.normal((10, latent_dim))
    fake_samples = best_generator.predict([label, noise])
    clinical_part = fake_samples[:, :clinical_data.shape[1]]
    gene_part = fake_samples[:, clinical_data.shape[1]:]
    decoded_clinical = encoder.inverse_transform(np.round(clinical_part).clip(min=0))
    combined = np.hstack([decoded_clinical, gene_part])
    df = pd.DataFrame(combined, columns=all_cols)
    stage_name = label_encoder.inverse_transform([i])[0].replace(' ', '_')
    df.to_csv(f"synthetic_by_stage/{stage_name}.csv", index=False)

print("✅ Final decoded samples saved per stage to synthetic_by_stage/")
