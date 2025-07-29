import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)

# ✅ Load model
MODEL_PATH = "ecg_model.h5"  # model in same folder
model = load_model(MODEL_PATH)

# ✅ Define class labels
class_names = ['Abnormal', 'Normal']

# ✅ Predict ECG
def preprocess(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ✅ Get form data
        fname = request.form.get("fname")
        lname = request.form.get("lname")
        age = request.form.get("age")
        gender = request.form.get("gender")

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files["file"]
        img = Image.open(file.stream).convert("RGB")
        processed = preprocess(img)

        # Predict
        prediction = model.predict(processed)
        predicted_index = int(np.argmax(prediction))
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction))

        #  Sample issues if abnormal
        problems = []
        if predicted_class == "Abnormal":
            problems = ["P1: Irregular Rhythm", "P2: ST Deviation", "P3: Tachycardia"]

        return jsonify({
            "first_name": fname,
            "last_name": lname,
            "age": age,
            "gender": gender,
            "prediction": predicted_class,
            "confidence": f"{confidence:.2f}",
            "problems": problems
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "ECG Classification API is running successfully "

if __name__ == "__main__":
    app.run()
