import streamlit as st
import numpy as np
import tempfile
from audio_recorder_streamlit import audio_recorder

st.set_page_config(page_title="Real-time Noisy Environment Sound Classifier", layout="centered")
st.title("Real-time Noisy Environment Sound Classifier")
st.write("""
Identify background sounds such as **speech**, **music**, **traffic**, or **silence** using audio captured from a microphone or uploaded file.
""")

# Input options
audio_source = st.radio("Choose audio input method:", ("Upload Audio File", "Record Audio"))
audio_data = None

if audio_source == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"]) 
    if uploaded_file is not None:
        audio_data = uploaded_file.read()
        st.audio(audio_data, format=uploaded_file.type)

elif audio_source == "Record Audio":
    st.write("Click the microphone button to start recording:")
    audio_bytes = audio_recorder()
    if audio_bytes is not None:
        audio_data = audio_bytes
        st.audio(audio_data, format="audio/wav")

# Placeholder for model prediction
def classify_audio(audio_bytes):
    # TODO: Replace with actual model inference
    # For now, return a dummy prediction
    return np.random.choice(["speech", "music", "traffic", "silence"])

if audio_data is not None:
    st.info("Classifying audio...")
    prediction = classify_audio(audio_data)
    st.success(f"Predicted Sound Type: **{prediction}**")
