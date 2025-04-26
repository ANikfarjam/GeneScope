import keras
from tensorflow.keras.models import load_model

# Enable unsafe deserialization (for Lambda layers in fusion model)
keras.config.enable_unsafe_deserialization()

# === Load models ===
clinical_model = load_model('./best_models1/best_clinical_model.keras')
gene_model = load_model('./best_models1/best_gene_model.keras')
fusion_model = load_model('./best_models1/fusion_model.keras')

# === Print model summaries ===
print("\n📋 Clinical Model Summary:")
clinical_model.summary()

print("\n📋 Gene Expression Model Summary:")
gene_model.summary()

print("\n📋 Fusion Model Summary:")
fusion_model.summary()
