import io
import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model(r'C:\Users\zaidl\OneDrive\Desktop\DoctorHair\DoctorHair\Code\HLprediction.h5')
IMG_SIZE = (80, 80)
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    img = image.convert('RGB').resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the POST request
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if not allowed_file(file.filename):
        return render_template('error.html', message='Invalid file type.')
    # Load the image and preprocess it
    img = Image.open(io.BytesIO(file.read()))
    img_array = preprocess_image(img)
    # Make a prediction
    prediction = model.predict(img_array)
    # Get the predicted label and display the result
    if prediction[0][0] > 0.5:
        label = 'Alopecia Areata'
        return render_template('result_aa.html', label=label)
    else:
        label = 'Healthy Hair'
        return render_template('result_hh.html', label=label)

if __name__ == '__main__':
    app.run(debug=True)
