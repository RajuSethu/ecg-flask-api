import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)

model = load_model("ecg_model.h5")
class_names = ['Abnormal', 'Normal']

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/')
def home():
    return "ECG Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = Image.open(file.stream).convert("RGB")
    processed = preprocess_image(img)
    prediction = model.predict(processed)
    predicted_class = class_names[int(np.argmax(prediction))]
    confidence = float(np.max(prediction))

    return jsonify({
        'prediction': predicted_class,
        'confidence': f"{confidence:.2f}"
    })
