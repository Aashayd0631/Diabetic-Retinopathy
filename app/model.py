import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mse
from PIL import Image

# Define the paths to the trained model and image folders 
MODEL_PATH = 'models/your_model.h5'  # Correct path to the pre-trained model
IMAGE_PATH = './models/dataset/'
'''
# Define the class labels for diabetic retinopathy severity
CLASS_LABELS = ['Healthy', 'Mild DR', 'Moderate DR', 'Proliferate DR', 'Severe DR']
'''

# Load the trained model
def load_trained_model():
    model = load_model(MODEL_PATH)
    return model

# Preprocess the uploaded images
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize the image to match model
    image = image / 255.0  # Normalize pixel values to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Make predictions using the trained model
'''def predict_diabetic_retinopathy(image):
    model = load_trained_model()
    image = preprocess_image(image)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    severity = CLASS_LABELS[predicted_class]
    return severity'''
# model.py

# Modify the CLASS_LABELS to reflect the desired output
CLASS_LABELS = ['NEGATIVE DR', 'POSITIVE DR']

# Modify the predict_diabetic_retinopathy function to return 'POSITIVE DR' or 'NEGATIVE DR'
def predict_diabetic_retinopathy(image):
    model = load_trained_model()
    image = preprocess_image(image)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    severity = CLASS_LABELS[predicted_class]
    return severity

