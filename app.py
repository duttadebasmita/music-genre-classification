import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from utils import load_and_preprocess_data, model_prediction

# Load the pre-trained model
model = tf.keras.models.load_model('C:\\musictrain\\Trained_model.h5')

# Define class names
classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'metal', 'pop', 'reggae', 'rock']

# Custom CSS for background and other styles
st.markdown("""
    <style>
        .main {
            background: linear-gradient(to right, #2b5876, #4e4376);
            color: white;
        }
        .sidebar .sidebar-content {
            background: #1e1e2f;
        }
        h1, h2, h3, h4, h5, h6 {
            color: white;
        }
        .css-18e3th9 {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 15px;
        }
        .css-1d391kg {
            color: white;
        }
        .css-1d391kg:hover {
            background: #4e4376;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Genre Classification"])

# Main Page
if app_mode == "Home":
    st.header("MUSIC GENRE CLASSIFICATION SYSTEM")
    image_path = "C:\\musictrain\\home_pic.jpeg"  # Make sure this image exists in your directory
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Music Genre Classification System! 🎵🔍

    Our mission is to help you identify music genres efficiently. Upload an audio file, and our system will analyze it to detect the genre. Let's explore the world of music together!

    ### How It Works
    1. **Upload Audio:** Go to the **Genre Classification** page and upload an audio file.
    2. **Analysis:** Our system will process the audio using advanced algorithms to identify the genre.
    3. **Results:** View the results and enjoy discovering the genre of your music.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate genre detection.
    - **User-Friendly:** Simple and intuitive interface for a seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Genre Classification** page in the sidebar to upload an audio file and experience the power of our Music Genre Classification System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
        ### Music Genre Classification Project
        This project is designed to classify music genres using machine learning. The model has been trained on a diverse dataset of music tracks, and it can recognize multiple genres with high accuracy.

        ### Project Features
        - **Dataset**: The model is trained on a comprehensive dataset of various music genres.
        - **Model Architecture**: We use state-of-the-art deep learning models to ensure accurate genre classification.
        - **User Interface**: The application provides an easy-to-use interface for uploading audio files and viewing results.

        ### Future Enhancements
        - **Real-time Analysis**: Implement real-time genre classification.
        - **Expanded Genres**: Add support for more music genres.
        - **Improved Accuracy**: Continuously improve model accuracy with more training data.
    """)

# Genre Classification Page
elif app_mode == "Genre Classification":
    st.header("Genre Classification")
    st.markdown("### Upload an audio file to classify its genre:")
    audio_file = st.file_uploader("Choose an Audio File:", type=["wav", "mp3"])

    if audio_file:
        # Save the uploaded file temporarily
        with open("temp_audio." + audio_file.name.split('.')[-1], "wb") as f:
            f.write(audio_file.getbuffer())

        audio_path = "temp_audio." + audio_file.name.split('.')[-1]
        st.audio(audio_file, format='audio/wav')

        # Predict button
        if st.button("Predict"):
            with st.spinner('Analyzing the audio file...'):
                X_test = load_and_preprocess_data(audio_path)
                genre_index = model_prediction(X_test, model)
                genre = classes[genre_index]
            st.success(f"Model prediction: **{genre.replace('_', ' ')}**")
            st.balloons()
            st.markdown(f"""
                ### Enjoy the Music!
                - The genre of the uploaded audio file is classified as **{genre.replace('_', ' ')}**.
                - Explore more music of this genre and enjoy!
            """)
