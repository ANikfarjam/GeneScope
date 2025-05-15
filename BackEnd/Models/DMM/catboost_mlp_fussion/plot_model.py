from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Load your best fusion model (adjust path as needed)
fusion_model = load_model("./best_models1/best_fusion_model.keras")

# Save a visual representation of the model
plot_model(
    fusion_model,
    to_file="fusion_model_architecture2.png",
    show_shapes=True,
    show_layer_names=True,
    expand_nested=True,
    dpi=150
)