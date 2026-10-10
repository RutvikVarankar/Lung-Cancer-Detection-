"""Pulmo-Scan AI - Flask backend for the VGG16 lung CT classifier."""
import io
import os
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # hides the oneDNN startup notice
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hides TensorFlow info/warning spam

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from keras import layers, models
from keras.applications import VGG16

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "lung_cancer_vgg16_model.keras"
HTML_FILE = "lung.html"
IMAGE_SIZE = (224, 224)

# IMPORTANT: must match the "Class index mapping" printed in the notebook (cell 8).
# image_dataset_from_directory sorts the class folders alphabetically.
CLASS_NAMES = ["Benign", "Malignant", "Normal"]

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB, same limit the page shows

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
CORS(app)  # lets the page work even if lung.html is opened straight from disk


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
def build_model() -> tf.keras.Model:
    """Identical architecture to the notebook (cell 11). Used as a fallback when the
    saved .keras file cannot be deserialized because of a Keras version mismatch."""
    base_model = VGG16(weights=None, include_top=False,
                       input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    return models.Sequential([
        layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.Rescaling(1.0 / 255),
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ])


def load_trained_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Copy lung_cancer_vgg16_model.keras into the same folder as app.py."
        )

    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as exc:  # typically a Keras version mismatch
        print(f"load_model failed ({type(exc).__name__}). "
              "Rebuilding the architecture and loading weights only...")

    model = build_model()
    model.load_weights(MODEL_PATH)
    return model


print("Loading model (this can take a minute)...")
model = load_trained_model()

if model.output_shape[-1] != len(CLASS_NAMES):
    raise ValueError(
        f"Model has {model.output_shape[-1]} outputs but CLASS_NAMES has {len(CLASS_NAMES)}."
    )

# Warm-up so the first real request is fast
model.predict(np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32), verbose=0)
print("Model ready. Classes:", CLASS_NAMES)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def preprocess(file_bytes: bytes) -> np.ndarray:
    """Same pipeline as training: RGB, bilinear resize to 224x224, raw 0-255 values.
    The model has a Rescaling(1/255) layer inside, so do NOT divide by 255 here."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    arr = tf.image.resize(arr, IMAGE_SIZE).numpy()  # bilinear, like image_dataset_from_directory
    return np.expand_dims(arr, axis=0)


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, HTML_FILE)


@app.route("/health")
def health():
    return jsonify(status="ok", classes=CLASS_NAMES)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify(error="No image uploaded."), 400
    if Path(file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        return jsonify(error="Unsupported file type. Please upload a PNG or JPG image."), 400

    try:
        batch = preprocess(file.read())
    except (UnidentifiedImageError, OSError):
        return jsonify(error="Could not read the file as an image."), 400

    probs = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    return jsonify({
        "class": CLASS_NAMES[idx],
        "confidence": round(float(probs[idx]) * 100, 2),
        "probabilities": {n: round(float(p) * 100, 2) for n, p in zip(CLASS_NAMES, probs)},
    })


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image is larger than 10 MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)