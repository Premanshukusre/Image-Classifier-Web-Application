import os
import datetime
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
try:
    import keras
except ImportError:
    from tensorflow import keras

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
]

CLASS_DISPLAY_NAMES = {
    "Apple___Apple_scab": "Apple Scab",
    "Apple___Black_rot": "Black Rot",
    "Apple___Cedar_apple_rust": "Cedar Apple Rust",
    "Apple___healthy": "Healthy Apple Leaf"
}

# Rich Botanical & Disease Intelligence Database
DISEASE_KNOWLEDGE_BASE = {
    "Apple___Apple_scab": {
        "display_name": "Apple Scab",
        "status": "warning",
        "severity_label": "Moderate to High Risk",
        "pathogen": "Venturia inaequalia (Fungal Infection)",
        "summary": "Apple scab is a serious fungal disease caused by Venturia inaequalis. It attacks leaves and fruit, causing brownish-olive spots and premature defoliation.",
        "symptoms": [
            "Olive-green to velvety brown velvety spots on leaf surfaces.",
            "Yellowing leaves followed by early leaf drop (defoliation).",
            "Crater-like corky lesions on fruit reducing marketability.",
            "Deformed growth on young leaf shoots."
        ],
        "organic_remedies": [
            "Apply liquid copper soap or sulfur sprays early in the season.",
            "Spray Neem oil solution at 7-14 day intervals during moist weather.",
            "Apply bio-fungicides containing Bacillus subtilis to prevent spore germination."
        ],
        "chemical_controls": [
            "Myclobutanil or Captan sprays applied at green tip stage.",
            "Mancozeb or Difenoconazole fungicide rotation to avoid pathogen resistance."
        ],
        "prevention_tips": [
            "Rake and destroy fallen infected leaves before winter.",
            "Prune tree canopy to maximize sun exposure and air circulation.",
            "Avoid overhead irrigation; keep foliage dry."
        ]
    },
    "Apple___Black_rot": {
        "display_name": "Black Rot",
        "status": "danger",
        "severity_label": "High Severity Risk",
        "pathogen": "Botryosphaeria obtusa (Fungal Pathogen)",
        "summary": "Black rot is a severe fungal disease causing 'frog-eye' leaf spots, cankers on branches, and black, shriveled mummified fruit on the tree.",
        "symptoms": [
            "Purple spots on leaves expanding into circular 'frog-eye' lesions with brown centers.",
            "Dark, sunken bark cankers on twigs and main limbs.",
            "Fruit rots into firm black mummy rings late in the season."
        ],
        "organic_remedies": [
            "Prune and destroy all dead twigs, cankers, and mummified fruits.",
            "Apply Bordeaux mixture (copper sulfate + lime) prior to bud break.",
            "Use bio-fungicide sprays with Trichoderma harzianum."
        ],
        "chemical_controls": [
            "Captan combined with Ziram applied from petal fall through harvest.",
            "Thiophanate-methyl or Pyraclostrobin for targeted canker suppression."
        ],
        "prevention_tips": [
            "Remove all mummified apples from trees and orchard floor before spring.",
            "Keep trees pruned and healthy through balanced nitrogen fertilization.",
            "Sanitize pruning equipment with 70% isopropyl alcohol between cuts."
        ]
    },
    "Apple___Cedar_apple_rust": {
        "display_name": "Cedar Apple Rust",
        "status": "warning",
        "severity_label": "Moderate Pathogen Severity",
        "pathogen": "Gymnosporangium juniperi-virginianae (Heteroecious Rust Fungus)",
        "summary": "Cedar Apple Rust requires two host plants (apple trees and Eastern Red Cedar) to complete its lifecycle, forming striking bright orange spots on leaves.",
        "symptoms": [
            "Bright yellow-orange spots on upper leaf surfaces appearing in spring.",
            "Small black fruiting bodies (pycnia) inside the orange spots.",
            "Fringe-like tube structures (aecia) on the underside of leaves.",
            "Premature defoliation resulting in weakened trees and smaller fruit."
        ],
        "organic_remedies": [
            "Apply sulfur or copper formulations at blossom time.",
            "Spray bio-fungicides at bud break to prevent rust spore penetration."
        ],
        "chemical_controls": [
            "Sterol-inhibiting fungicides (e.g., Myclobutanil, Propiconazole).",
            "Mancozeb or Fenbuconazole starting from pink bud stage through early summer."
        ],
        "prevention_tips": [
            "Remove Eastern Red Cedar / Juniper trees within a 1-2 mile radius if feasible.",
            "Plant rust-resistant apple cultivars (e.g., Liberty, Enterprise, Freedom).",
            "Apply preventive sprays during rainy periods between pink bud and petal fall."
        ]
    },
    "Apple___healthy": {
        "display_name": "Healthy Apple Leaf",
        "status": "healthy",
        "severity_label": "Optimal Health",
        "pathogen": "None detected",
        "summary": "The leaf shows optimal chloroplast activity, uniform deep green pigmentation, intact cell structures, and no symptoms of fungal or bacterial pathogens.",
        "symptoms": [
            "Vibrant, deep green color across upper and lower epidermis.",
            "Clear, unblemished leaf margins and smooth venation network.",
            "Robust turgor pressure and healthy leaf tissue structure."
        ],
        "organic_remedies": [
            "Maintain standard organic compost tea or kelp meal leaf sprays to boost natural immunity.",
            "Mulch tree base with organic wood chips to preserve soil moisture."
        ],
        "chemical_controls": [
            "No chemical intervention required."
        ],
        "prevention_tips": [
            "Continue routine orchard monitoring and proper drip irrigation.",
            "Maintain balanced N-P-K soil nutrients based on annual soil tests.",
            "Inspect foliage bi-weekly during humid weather conditions."
        ]
    }
}

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "apple_disease_model.keras"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
SAMPLES_DIR = PROJECT_ROOT / "static" / "samples"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("LOADING APPLE DISEASE AI MODEL")
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


print(f"Model loaded successfully from: {MODEL_PATH}")

# ============================================================
# PREDICTION HELPER
# ============================================================

def predict_image(image_path):
    """
    Load an image and compute predictions across all 4 classes.
    Returns:
        predicted_class (str): raw class name
        confidence (float): highest probability as percentage (0-100)
        all_probabilities (dict): mapping of display names to float percentages
    """
    image = keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    image_array = keras.utils.img_to_array(image)
    image_array = image_array / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index] * 100)

    all_probabilities = {}
    for idx, raw_class in enumerate(CLASS_NAMES):
        disp_name = CLASS_DISPLAY_NAMES[raw_class]
        prob_pct = float(predictions[idx] * 100)
        all_probabilities[disp_name] = round(prob_pct, 2)

    return predicted_class, confidence, all_probabilities

# ============================================================
# STATIC ROUTES FOR UPLOADS & SAMPLES
# ============================================================

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOADS_DIR, filename)

@app.route("/static/samples/<path:filename>")
def serve_sample(filename):
    return send_from_directory(SAMPLES_DIR, filename)

# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    all_probs = None
    disease_info = None
    image_url = None
    error = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "No image file provided."
        else:
            file = request.files["image"]
            if file.filename == "":
                error = "Please select an image file to upload."
            else:
                try:
                    # Save uploaded file safely with timestamp
                    time_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
                    safe_filename = time_prefix + file.filename.replace(" ", "_")
                    image_path = UPLOADS_DIR / safe_filename
                    file.save(image_path)

                    pred_class, conf, all_probs = predict_image(image_path)
                    prediction = CLASS_DISPLAY_NAMES.get(pred_class, pred_class)
                    confidence = conf
                    disease_info = DISEASE_KNOWLEDGE_BASE.get(pred_class, {})
                    image_url = f"/uploads/{safe_filename}"
                except Exception as e:
                    error = f"Prediction failed: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        all_probabilities=all_probs,
        disease_info=disease_info,
        image_url=image_url,
        error=error,
        samples=[
            {"id": "apple_scab", "name": "Apple Scab", "file": "apple_scab.jpg"},
            {"id": "black_rot", "name": "Black Rot", "file": "black_rot.jpg"},
            {"id": "cedar_rust", "name": "Cedar Apple Rust", "file": "cedar_apple_rust.jpg"},
            {"id": "healthy", "name": "Healthy Leaf", "file": "healthy.jpg"}
        ]
    )

# ============================================================
# API ENDPOINT FOR AJAX / WEBCAM / SAMPLES
# ============================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        image_path = None
        filename_used = ""
        image_url = ""

        # Case 1: Preset Sample selected
        if request.is_json and "sample" in request.json:
            sample_file = request.json["sample"]
            sample_path = SAMPLES_DIR / sample_file
            if not sample_path.exists():
                return jsonify({"success": False, "error": f"Sample image '{sample_file}' not found."}), 404
            
            # Copy sample into uploads for serving
            time_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
            filename_used = time_prefix + sample_file
            image_path = UPLOADS_DIR / filename_used
            with open(sample_path, "rb") as sf, open(image_path, "wb") as df:
                df.write(sf.read())
            image_url = f"/uploads/{filename_used}"

        # Case 2: File upload via Form Data
        elif "image" in request.files:
            file = request.files["image"]
            if file.filename == "":
                return jsonify({"success": False, "error": "No selected image file."}), 400

            time_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
            filename_used = time_prefix + file.filename.replace(" ", "_")
            image_path = UPLOADS_DIR / filename_used
            file.save(image_path)
            image_url = f"/uploads/{filename_used}"

        else:
            return jsonify({"success": False, "error": "Invalid request payload. Send file or sample."}), 400

        # Perform prediction
        pred_class, conf, all_probs = predict_image(image_path)
        disease_info = DISEASE_KNOWLEDGE_BASE.get(pred_class, {})
        disp_name = CLASS_DISPLAY_NAMES.get(pred_class, pred_class)

        timestamp_str = datetime.datetime.now().strftime("%b %d, %Y - %I:%M %p")

        return jsonify({
            "success": True,
            "raw_class": pred_class,
            "prediction": disp_name,
            "confidence": round(conf, 2),
            "all_probabilities": all_probs,
            "disease_info": disease_info,
            "image_url": image_url,
            "filename": filename_used,
            "timestamp": timestamp_str
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"Internal Server Error: {str(e)}"}), 500

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STARTING ADVANCED APPLE DISEASE CLASSIFIER AGENT")
    print("=" * 70)
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() in ["true", "1"]
    app.run(host="0.0.0.0", port=port, debug=debug_mode)