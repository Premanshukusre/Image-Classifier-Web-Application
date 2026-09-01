from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
]


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "apple_disease_model.keras"


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    print("=" * 70)
    print("LOADING APPLE DISEASE CLASSIFICATION MODEL")
    print("=" * 70)

    print(f"\nModel path: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = keras.models.load_model(MODEL_PATH)

    print("\nModel loaded successfully.")

    return model


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(model, image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    print("\n" + "=" * 70)
    print("PROCESSING IMAGE")
    print("=" * 70)

    print(f"\nImage: {image_path}")

    # Load image
    image = keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    # Convert image to NumPy array
    image_array = keras.utils.img_to_array(image)

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    print(f"Image shape: {image_array.shape}")

    # Generate prediction
    predictions = model.predict(image_array, verbose=0)

    # Get predicted class
    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]

    # Get confidence
    confidence = predictions[0][predicted_index] * 100

    print("\n" + "=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)

    print(f"\nPredicted class : {predicted_class}")
    print(f"Confidence      : {confidence:.2f}%")

    print("\nClass probabilities:")

    for index, class_name in enumerate(CLASS_NAMES):
        probability = predictions[0][index] * 100
        print(f"  {class_name:<35} {probability:6.2f}%")

    print("\n" + "=" * 70)

    return predicted_class, confidence


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    model = load_model()

    image_path = input(
        "\nEnter path to apple leaf image: "
    ).strip().strip('"')

    predict_image(model, image_path)