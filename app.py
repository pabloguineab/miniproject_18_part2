import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import pyjokes

# Set page title and layout
st.set_page_config(
    page_title="AI Image Recognizer",
    layout="wide"
)

# Define the three models
model_dict = {
    "MobileNetV2": MobileNetV2,
    "ResNet50": ResNet50,
    "VGG16": VGG16
}

# Function to load model
@st.cache(allow_output_mutation=True)
def load_model(model_name):
    model_class = model_dict[model_name]
    model = model_class(weights="imagenet")
    return model

# Define image size
IMAGE_SIZE = (224, 224)

# Function to make predictions
def predict(image, model, top_k=3):
    st.info("Resizing image...")
    image = ImageOps.fit(image, IMAGE_SIZE, Image.LANCZOS)
    image = np.asarray(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    st.info("Predicting image...")
    prediction = model.predict(image)
    decoded_predictions = decode_predictions(prediction, top=top_k)[0]
    return decoded_predictions

# Function to get a random joke
def get_random_joke():
    return pyjokes.get_joke()

# Function for UI layout
def run_app():
    st.sidebar.image("George_Brown_College_logo.svg.png", use_column_width=True)
    st.sidebar.header("About")
    st.sidebar.info(
        "This application is a demonstration of how to use "
        "pre-trained models for image classification tasks using Streamlit and TensorFlow. "
        "It uses various models, including MobileNetV2, ResNet50, and VGG16, which are trained on the ImageNet dataset. "
        "The app will predict the class of the uploaded image out of 1000 classes. "
    )
    
    st.sidebar.header("Team Members")
    st.sidebar.text(
        """
        - Pablo Guinea Benito
        - Joy
        - Abdullah
        - Sebastian
        """
    )
    
    # Create a selectbox widget to choose the model
    st.sidebar.title("Select Model")
    st.sidebar.markdown("Choose the model to use for image recognition:")
    model_name = st.sidebar.selectbox("", list(model_dict.keys()))

    model = load_model(model_name)

    st.title("AI Image Recognizer")
    st.header("Predict the class of an uploaded image")

    file = st.file_uploader("Please upload an image", type=["jpg", "png", "jpeg"])