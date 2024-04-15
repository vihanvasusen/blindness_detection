from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from flask import send_from_directory

app = Flask(__name__)

# Load the pre-trained model
model = load_model(r'C:\Users\Mobile Programming\Desktop\binary dr\binary_dr_wgt_model.hdf5')
model.make_predict_function()

# Define the upload folder
UPLOAD_FOLDER = r'C:\Users\Mobile Programming\Desktop\binary dr\static\user_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed extensions for file uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# This route is responsible for serving uploaded images
@app.route('/static/user_uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return render_template('index.html', prediction=None, confidence=None, filename=None)

        if file and allowed_file(file.filename):
            # Save the file
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Make a prediction
            prediction, confidence = predict_image(filename)

            return render_template('index.html', prediction=prediction, confidence=confidence, filename=file.filename)

    return render_template('index.html', prediction=None, confidence=None, filename=None)

def predict_image(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    confidence = prediction[0][0]  # Assuming it's a binary classification, adjust accordingly for multi-class

    if confidence > 0.5:
        result = 'Diabetic Retinopathy (DR)'
    else:
        result = 'No Diabetic Retinopathy (No DR)'

    return result, f'Confidence: {confidence:.2%}'

if __name__ == '__main__':
    app.run(debug=True)
