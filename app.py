from flask import Flask, render_template, redirect, request, send_from_directory
from tensorflow.keras import models, optimizers, losses
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
import os
from PIL import Image
import numpy as np
import cv2



app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the best deep learning model
model_file = "BO_model.h5"
model = models.load_model(model_file, compile=False)
model.compile(optimizer=optimizers.Adam(lr=0.001),
             loss=losses.BinaryCrossentropy(),
             metrics=["accuracy"])
print("Model loaded")


def make_predictions(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (128,128))
    x = image.img_to_array(img_resized)
    x_arr = np.expand_dims(x, axis=0)
    x_input = preprocess_input(x_arr, mode='tf')

    predict = model.predict(x_input)
    p = predict[0,0]
    if p > 0.5:
        p = "Pneumonic"
    else:
        p = "Healthy"
    return p


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check the uploaded file
        if 'img' not in request.files:
            return render_template('home.html', filename="upload_img.jpg", message="Upload an image")

        f = request.files['img']
        if f.filename == '':
            return render_template('home.html', filename="upload_img.jpg", message="No image selected")

        if not ('jpeg' in f.filename or 'png' in f.filename or 'jpg' in f.filename):
            return render_template('home.html', filename="upload_img.jpg",
                                   message="Only .png or .jpg/.jpeg images are allowed")

        # Save image to upload folder and make sure only 1 file exists
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        if len(files) == 1:
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        else:
            files.remove("upload_img.jpg")
            file_ = files[0]
            os.remove(app.config['UPLOAD_FOLDER'] + '/' + file_)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))

        predictions = make_predictions(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        return render_template('home.html', filename=f.filename, message=predictions, show=True)
    return render_template('home.html', filename="upload_img.jpg")


if __name__ == "__main__":
    app.run(debug=True)
