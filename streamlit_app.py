import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import soundfile as sf
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av

# Load model and encoder
model = tf.keras.models.load_model('sentiment_cnn_model.h5')
le = pickle.load(open('le.pkl', 'rb'))

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def predict_sentiment(file_path):
    features = extract_features(file_path).reshape(1, 40, 1, 1)
    prediction = model.predict(features)
    label = le.inverse_transform([np.argmax(prediction)])[0]
    confidence = np.max(prediction) * 100
    return label, confidence

st.set_page_config(page_title="Audio Sentiment", layout="centered")
st.title("🎧 Audio Sentiment Analysis using CNN")

# === File Upload ===
st.header("📤 Upload Audio File")
uploaded_file = st.file_uploader("Choose a WAV or MP3 file", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        sentiment, conf = predict_sentiment(tmp_path)
        st.success(f"🎯 Predicted Sentiment: `{sentiment}` ({conf:.2f}% confidence)")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# === Audio Recording ===
st.header("🎙️ Record Audio in Browser")

# Use session state to store audio frames
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = []

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recorded_frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten()
        self.recorded_frames.append(audio)
        return frame

    def get_frames(self):
        return self.recorded_frames

    def reset(self):
        self.recorded_frames = []

# Create processor
processor_instance = AudioProcessor()

webrtc_ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=lambda: processor_instance,
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

if not webrtc_ctx.state.playing and processor_instance.get_frames():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        audio_data = np.concatenate(processor_instance.get_frames())
        sf.write(tmp_file.name, audio_data, 48000)  # 48kHz sample rate

        st.success("✅ Recording complete.")
        st.audio(tmp_file.name, format="audio/wav")

        try:
            sentiment, conf = predict_sentiment(tmp_file.name)
            st.success(f"🎯 Predicted Sentiment: `{sentiment}` ({conf:.2f}% confidence)")
        except Exception as e:
            st.error(f"❌ Error processing recorded audio: {e}")

    processor_instance.reset()
