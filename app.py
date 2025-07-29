from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load model safely
try:
    model = load_model("ecg_model.h5")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Health check route (for Railway or uptime bots)
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ ECG Classification API is running"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    try:
        image = Image.open(request.files["file"]).convert("RGB").resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image)[0]
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return jsonify({
            "class": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500
