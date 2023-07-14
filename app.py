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