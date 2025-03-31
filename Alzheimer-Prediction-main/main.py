import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('model.h5', compile=False)  # Ensure compatibility

# Define the class names (zero-indexed issue fixed)
class_names = {
    1: 'Mild Demented',
    2: 'Moderate Demented',
    3: 'Non Demented',
    4: 'Very Mild Demented'
}

img_height = 220
img_width = 220

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((img_width, img_height))  # Resize image
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Streamlit app header
st.title("Alzheimer's Prediction")

# User input
name = st.text_input("Enter your name:")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Display the uploaded image
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Predict button
if st.button("Predict") and uploaded_file:
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Predict the class probabilities
    predictions = model.predict(preprocessed_image)
    
    # Get the predicted class index (shifted to match dictionary)
    predicted_class_index = np.argmax(predictions[0]) + 1
    
    # Get the predicted class label
    predicted_class_label = class_names[predicted_class_index]
    
    # Display the prediction
    st.success(f"Predicted Class: {predicted_class_label}")
elif st.button("Predict"):
    st.warning("Please upload an image.")
