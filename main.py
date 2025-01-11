import streamlit as st
import tensorflow
from PIL import Image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalMaxPooling2D
import cv2
import numpy as np
from numpy.linalg import norm
import pickle
from sklearn.neighbors import NearestNeighbors
import os

# Load feature vectors and filenames
feature_list = np.array(pickle.load(open("featurevector.pkl", "rb")))
filenames = pickle.load(open("filenames.pkl", "rb"))

# Load the ResNet50 model without the top layers (include_top=False) and add pooling layer
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
model.summary()

# Streamlit app title
st.title('VISUAL SEARCH')

# Function to save uploaded file
def save_upload_file(upload_file):
    try:
        with open(os.path.join('uploads', upload_file.name), 'wb') as f:
            f.write(upload_file.getbuffer())
        return 1
    except:
        return 0

# Function to extract features from the uploaded image
def extract_feature(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  # Resizing to 224x224 for ResNet50
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Recommendation function using NearestNeighbors
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File upload widget
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_upload_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        resized_img = display_image.resize((200, 200))
        st.image(resized_img)

        # Extract features of the uploaded image
        features = extract_feature(os.path.join("uploads", uploaded_file.name), model)

        # Get recommendations based on the extracted features
        indices = recommend(features, feature_list)

        # Display recommended images
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(filenames[indices[0][0]])  # Show the first recommended image
        with col2:
            st.image(filenames[indices[0][1]])  # Show the second recommended image
        with col3:
            st.image(filenames[indices[0][2]])  # Show the third recommended image
        with col4:
            st.image(filenames[indices[0][3]])  # Show the fourth recommended image
        with col5:
            st.image(filenames[indices[0][4]])  # Show the fifth recommended image
    else:
        st.header("Some error has occurred during file upload")
