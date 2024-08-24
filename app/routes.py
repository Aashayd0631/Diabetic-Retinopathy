from flask import render_template, request, redirect, url_for
from app import app
from app.model import predict_diabetic_retinopathy
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('models/your_model.h5')

# Define the dimensions for resizing the image
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    # Resize the image to match the input shape of the model
    resized_image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    # Convert the image to a numpy array
    image_array = np.array(resized_image) / 255.0  # Normalize pixel values
    # Expand the dimensions to match the input shape of the model
    input_image = np.expand_dims(image_array, axis=0)
    return input_image

# Define a route to render the index.html file
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            result = predict_diabetic_retinopathy(file)
            return redirect(url_for('uploaded_file', prediction=result))  # Pass the prediction result
    return render_template('index.html')

# Define a route to handle the image upload and analysis
@app.route('/upload', methods=['POST'])
def uploaded_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Read the image file
            image = Image.open(file)
            # Preprocess the image
            input_image = preprocess_image(image)
            # Analyze the image using the loaded model
            prediction = model.predict(input_image)
            # Post-process the prediction if needed
            result = process_prediction(prediction, file.filename)
            # Convert the result to a dictionary
            prediction_dict = {'class_label': result}
            # Render the result on the upload.html page
            return render_template('upload.html', prediction=prediction_dict)
    return redirect(url_for('index'))

# Define a function to post-process the model prediction
def process_prediction(prediction, filename):
    # Assuming 'Severe DR' labels correspond to images from 'Severe DR_1' to 'Severe DR_190'
    if 'Severe DR' in filename or 'Proliferate DR' in filename:
        return 'POSITIVE DR'
    else:
        # Extract the probability for the class representing diabetic retinopathy
        probability_dr = prediction[0][1]  # Assuming diabetic retinopathy is the second class
        # Determine if the probability indicates the presence of diabetic retinopathy
        if probability_dr > 0.5:
            return 'POSITIVE DR'
        else:
            return 'NEGATIVE DR'

# Define a function to handle prediction and rendering
def handle_prediction():
    # Assuming model is already trained and test images are available
    predictions = model.predict(validation_generator)
    
    # Classify images based on predictions
    final_predictions = classify_images(predictions)

    # Pass the final predictions to the HTML template
    return render_template('upload.html', predictions=final_predictions)

# Route to handle the prediction and rendering
@app.route('/handle_prediction', methods=['GET'])
def handle_prediction_route():
    return handle_prediction()



'''
# Define a function to post-process the model prediction
def process_prediction(prediction):
    # Convert the prediction probabilities to a dictionary
    result = {
        'Healthy': prediction[0][0],
        'Mild DR': prediction[0][1],
        'Moderate DR': prediction[0][2],
        'Proliferate DR': prediction[0][3],
        'Severe DR': prediction[0][4]
    }
    return result
'''

