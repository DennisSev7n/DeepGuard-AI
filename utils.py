import io
import time
import tempfile
import os
from PIL import Image
import torch
from transformers import pipeline
import librosa
import soundfile as sf
import streamlit as st

# Model IDs
IMAGE_MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
AUDIO_MODEL_ID = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"

# ---------------------------
# Cached models
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_image_pipe():
    return pipeline("image-classification", model=IMAGE_MODEL_ID,
                    device=0 if torch.cuda.is_available() else -1)

@st.cache_resource(show_spinner=False)
def get_audio_pipe():
    return pipeline("audio-classification", model=AUDIO_MODEL_ID,
                    device=0 if torch.cuda.is_available() else -1)

@st.cache_resource(show_spinner=False)
def get_text_detector():
    return pipeline("text-classification", model="roberta-large-openai-detector")

# ---------------------------
# Audio analysis
# ---------------------------
def process_audio_file(audio_bytes, filename):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            y, sr = librosa.load(tmp_path, sr=16000, mono=True)
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, y, sr, format="WAV")
            wav_buffer.seek(0)

            pipe = get_audio_pipe()
            out = pipe(wav_buffer.read())

            label = out[0]["label"].lower()
            score = out[0]["score"] * 100.0

            if "real" in label or "bonafide" in label:
                score = (1 - out[0]["score"]) * 100.0

            return score, out
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        return 0, {"error": f"Error processing audio: {str(e)}"}

# ---------------------------
# Text analysis
# ---------------------------
def text_score_ml(text):
    try:
        if len(text.strip()) < 50:
            return 0, {"error": "Text is too short for accurate analysis", "text_length": len(text)}

        detector = get_text_detector()
        out = detector(text[:1000])  # limit for performance

        label = out[0]["label"].lower()
        score = out[0]["score"] * 100.0

        if "fake" in label or "generated" in label:
            likelihood = score
        elif "real" in label or "human" in label:
            likelihood = 100 - score
        else:
            likelihood = score

        return likelihood, {"detector_label": label, "detector_confidence": score}
    except Exception as e:
        return 0, {"error": f"Error processing text: {str(e)}"}

# ---------------------------
# Image analysis
# ---------------------------
def image_score_ml(image_bytes):
    try:
        pipe = get_image_pipe()
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        out = pipe(pil)
        label = out[0]["label"].lower()
        score = out[0]["score"] * 100.0

        if "real" in label or "realism" in label:
            score = (1 - out[0]["score"]) * 100.0

        return score, out
    except Exception as e:
        return 0, {"error": f"Error processing image: {str(e)}"}

# ---------------------------
# Wrappers
# ---------------------------
def audio_score_ml(audio_bytes, filename):
    return process_audio_file(audio_bytes, filename)

def setup_session_state():
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "analysis_details" not in st.session_state:
        st.session_state.analysis_details = None

def analyze_content(modality, file_bytes, progress_bar=None, filename=None):
    for percent_complete in range(100):
        time.sleep(0.005)
        if progress_bar:
            progress_bar.progress(percent_complete + 1)

    if modality in ["Image", "Video"]:
        return image_score_ml(file_bytes)
    elif modality == "Audio":
        return audio_score_ml(file_bytes, filename)

    return 0, {"error": "Unsupported modality"}

def analyze_text(text, progress_bar=None):
    for percent_complete in range(100):
        time.sleep(0.005)
        if progress_bar:
            progress_bar.progress(percent_complete + 1)

    return text_score_ml(text)
