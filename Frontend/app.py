import streamlit as st
import numpy as np
import librosa
import pickle
import tempfile
from tensorflow.keras.models import load_model
from audio_recorder_streamlit import audio_recorder
from PIL import Image

# --------------------------
# 🌟 Page Configuration
# --------------------------
st.set_page_config(
    page_title="UrbanSound Classifier",
    page_icon="🎧",
    layout="centered",
)

# --------------------------
# 🎨 Custom CSS Styling
# --------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #1f1c2c, #928dab);
        color: white;
    }
    .stApp {
        background: linear-gradient(135deg, #232526, #414345);
        color: white;
    }
    .main-title {
        font-size: 2.2em;
        text-align: center;
        color: #00FFC6;
        font-weight: bold;
        margin-bottom: 0.2em;
    }
    .sub-text {
        text-align: center;
        color: #e0e0e0;
        margin-bottom: 2em;
    }
    .prediction {
        font-size: 1.5em;
        font-weight: bold;
        color: #00FFC6;
        text-align: center;
    }
    .upload-box {
        border: 2px dashed #00FFC6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: rgba(255,255,255,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# 🧩 Load Model and Encoder
# --------------------------
@st.cache_resource
def load_resources():
    model = load_model("../urbansound8k_model.h5")
    with open("../label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_resources()

# --------------------------
# 🔉 App Title
# --------------------------
st.markdown('<div class="main-title">🎧 Real-time Urban Sound Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Upload or record audio to identify environmental sounds in real-time.</div>', unsafe_allow_html=True)

# --------------------------
# 🎙️ Input Section
# --------------------------
audio_source = st.radio("Select Audio Input Method:", ["Upload Audio File", "Record Audio"], horizontal=True)
audio_data = None

if audio_source == "Upload Audio File":
    with st.container():
        st.markdown('<div class="upload-box">📤 Drag and drop an audio file (WAV, MP3, OGG)</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["wav", "mp3", "ogg"])
        if uploaded_file is not None:
            audio_data = uploaded_file.read()
            st.audio(audio_data, format=uploaded_file.type)

elif audio_source == "Record Audio":
    st.markdown('<div style="text-align:center;">🎙️ Click below to record:</div>', unsafe_allow_html=True)
    audio_bytes = audio_recorder()
    if audio_bytes is not None:
        audio_data = audio_bytes
        st.audio(audio_data, format="audio/wav")

# --------------------------
# 🧠 Prediction Function
# --------------------------
def predict_audio(audio_bytes):
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # Load and preprocess
    y, sr = librosa.load(tmp_path, duration=2.5, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0).reshape(1, 40)

    # Predict
    prediction = model.predict(mfccs_processed)
    predicted_label = le.classes_[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    return predicted_label, confidence

# --------------------------
# 🚀 Run Prediction
# --------------------------
if audio_data is not None:
    with st.spinner("🔍 Analyzing audio..."):
        label, conf = predict_audio(audio_data)
        st.markdown(f'<div class="prediction">🎯 Predicted Sound: <br><br> {label.upper()} <br><br>Confidence: {conf:.2f}%</div>', unsafe_allow_html=True)
