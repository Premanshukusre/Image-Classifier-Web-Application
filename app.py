from pathlib import Path

import numpy as np
from flask import Flask, render_template, request
try:
    import keras
except ImportError:
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
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "apple_disease_model.keras"


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("LOADING APPLE DISEASE MODEL")
print("=" * 70)

if not MODEL_PATH.exists():
    print(f"Model not found at {MODEL_PATH}. Building initial model...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    import sys
    sys.path.append(str(PROJECT_ROOT / "src"))
    from model import build_model
    model = build_model()
    model.save(MODEL_PATH)
else:
    model = keras.models.load_model(MODEL_PATH)


print(f"Model loaded from: {MODEL_PATH}")
print("Model loading completed successfully.")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):
    """
    Load an image and return the predicted class and confidence.
    """

    image = keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image_array = keras.utils.img_to_array(image)

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(predictions[0])

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(
        predictions[0][predicted_index] * 100
    )

    return predicted_class, confidence


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    error = None

    if request.method == "POST":

        if "image" not in request.files:
            error = "No image was uploaded."
            return render_template(
                "index.html",
                prediction=prediction,
                confidence=confidence,
                error=error
            )

        image = request.files["image"]

        if image.filename == "":
            error = "Please select an image."
            return render_template(
                "index.html",
                prediction=prediction,
                confidence=confidence,
                error=error
            )

        # Save uploaded image
        uploads_dir = PROJECT_ROOT / "uploads"
        uploads_dir.mkdir(exist_ok=True)

        image_path = uploads_dir / image.filename

        image.save(image_path)

        try:

            prediction, confidence = predict_image(
                image_path
            )

        except Exception as e:

            error = f"Prediction failed: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("STARTING APPLE DISEASE CLASSIFIER")
    print("=" * 70)

    app.run(
        debug=True
    )