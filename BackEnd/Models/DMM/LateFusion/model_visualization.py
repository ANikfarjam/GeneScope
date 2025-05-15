import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Enable Lambda deserialization (used in fusion layers)
keras.config.enable_unsafe_deserialization()

# Load the fusion model (which embeds gene + clinical models)
fusion_model = load_model("./best_models1/fusion_model.keras")

# Plot the entire architecture as one connected diagram
plot_model(
    fusion_model,
    to_file="fusion_model_composite.png",
    show_shapes=True,
    show_layer_names=True,
    expand_nested=True,  # <- Unfold gene/clinical internal layers
    dpi=100
)

print("✅ Saved full architecture diagram to 'fusion_model_composite.png'")
