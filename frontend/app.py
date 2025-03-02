import streamlit as st
import requests
import os

# Set page configuration
st.set_page_config(page_title="VibeCheckAI", page_icon="🎵", layout="wide")

# UI Styling
st.markdown(
    """
    <style>
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #04C6C9;
            text-align: center;
        }
        .subtitle {
            font-size: 24px;
            text-align: center;
            color: #D3615A;
        }
        .upload-box {
            border: 2px dashed #04C6C9;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin: auto;
            width: 60%;
        }
        .result {
            font-size: 30px;
            font-weight: bold;
            color: #D3615A;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display title
st.markdown('<div class="title">VibeCheckAI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">🎤 Giving Your Voice a Quick Mood Check 🎵</div>', unsafe_allow_html=True)
st.markdown("---")

# File uploader
st.markdown('<div class="upload-box">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an audio file (WAV/MP3)", type=["wav", "mp3"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    # Save file temporarily
    file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(file_path, format="audio/wav")

    # Send request to FastAPI
    with st.spinner("Analyzing..."):
        with open(file_path, "rb") as audio_file:
            files = {"file": (uploaded_file.name, audio_file, "audio/wav")}
            API_URL = "https://speechemotionrecognition.onrender.com"
            response = requests.post(API_URL, files=files)

    # Show result
    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted Emotion: {result['emotion']}")
    else:
        st.error(f"Error: {response.json()['detail']}")
