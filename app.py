from __future__ import division, print_function
import os
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
app = Flask('crack_detection')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'crack_detection.h5')
model = tf.keras.models.load_model(MODEL_PATH)
model.make_predict_function()
def predict(img, model):
    img = img.resize((128, 128))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    preds = model.predict(images)
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image_class():
    img = request.files['file'].read()
    img = Image.open(io.BytesIO(img))
    prediction = predict(img, model)
    class_name = "crack" if prediction[0] < 0.5 else "no_crack"
    response = {"prediction": class_name}
    return jsonify(response)

