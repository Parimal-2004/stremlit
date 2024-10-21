
import streamlit as st
import numpy as np
import librosa
import joblib

# Title of the app
st.title('Deepfake Voice Detection')

# Description
st.write("Upload an audio file (WAV or MP3) and the model will detect if it's a real or deepfake voice.")

# Upload audio file
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

# Process the uploaded file
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    # Load the audio file using librosa
    y, sr = librosa.load(uploaded_file, sr=None)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    
    # Load the pre-trained model
    model = joblib.load('/content/drive/MyDrive/path_to_your_model/model.joblib')
    
    # Make a prediction
    prediction = model.predict([mfccs])
    
    # Display the result
    if prediction == 0:
        st.success("The audio is Real.")
    else:
        st.error("The audio is a Deepfake.")
