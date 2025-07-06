# import streamlit as st
# import librosa
# import numpy as np
# import tensorflow as tf
# import pickle
# import soundfile as sf
# import tempfile

# # Load trained model and label encoder
# model = tf.keras.models.load_model("sentiment_cnn_model.h5")
# le = pickle.load(open("le.pkl", "rb"))

# st.set_page_config(page_title="Audio Sentiment Analysis", layout="centered")

# st.title("🎧 Audio Sentiment Detection using CNN")
# st.markdown("""
# Upload a spoken audio file and this app will analyze its **sentiment** using a trained CNN model.
# """)

# # Extract MFCCs
# def extract_features(audio, sr):
#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
#     return np.mean(mfcc.T, axis=0)

# # Predict function
# def predict(audio, sr):
#     feature = extract_features(audio, sr)
#     feature = feature.reshape(1, 40, 1, 1)
#     pred = model.predict(feature)
#     label = le.inverse_transform([np.argmax(pred)])[0]
#     return label

# # Upload audio
# audio_file = st.file_uploader("📤 Upload an audio file", type=["wav", "mp3", "m4a"])

# if audio_file is not None:
#     st.audio(audio_file, format="audio/wav")

#     # Save to a temp file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#         tmp.write(audio_file.read())
#         tmp_path = tmp.name

#     # Load and predict
#     audio, sr = librosa.load(tmp_path, sr=None)
#     label = predict(audio, sr)

#     st.success(f"🧠 Predicted Sentiment: **{label}**")
# --------------------------------------------------

# import streamlit as st
# import librosa
# import numpy as np
# import tensorflow as tf
# import pickle
# import tempfile
# from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
# #from streamlit_webrtc import WebRtcMode, ClientSettings
# import av
# import soundfile as sf
# from streamlit_webrtc import WebRtcMode


# # Load model & LabelEncoder
# model = tf.keras.models.load_model("sentiment_cnn_model.h5")
# le = pickle.load(open("le.pkl", "rb"))

# st.set_page_config(page_title="🎧 Audio Sentiment Analysis", layout="centered")

# st.title("🎤 Audio Sentiment Detection using CNN")

# # Feature extraction
# def extract_features(audio, sr):
#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
#     return np.mean(mfcc.T, axis=0)

# def predict(audio, sr):
#     feature = extract_features(audio, sr)
#     feature = feature.reshape(1, 40, 1, 1)
#     prediction = model.predict(feature)
#     label = le.inverse_transform([np.argmax(prediction)])[0]
#     return label

# # --------------------------------------------
# # Upload Section
# st.header("📤 Upload Audio File")
# audio_file = st.file_uploader("Choose a file", type=["wav", "mp3", "m4a"])

# if audio_file:
#     st.audio(audio_file, format="audio/wav")
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#         tmp.write(audio_file.read())
#         tmp_path = tmp.name

#     audio, sr = librosa.load(tmp_path, sr=None)
#     label = predict(audio, sr)
#     st.success(f"🧠 Predicted Sentiment: **{label}**")


# # Record Section
# st.header("🎙️ Or Record Your Voice")

# class AudioProcessor(AudioProcessorBase):
#     def __init__(self):
#         self.recorded_frames = []

#     def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
#         self.recorded_frames.append(frame)
#         return frame

# ctx = webrtc_streamer(
#     key="speech-recorder",
#     #mode="SENDRECV",
#     mode=WebRtcMode.SENDRECV,
#     audio_processor_factory=AudioProcessor
# )

# # if ctx.audio_receiver and ctx.audio_processor:
# #     frames = ctx.audio_processor.recorded_frames
# #     if len(frames) > 0:
# #         st.success("✅ Recording complete. Click below to analyze.")
# #         if st.button("🔍 Analyze Recorded Audio"):
# #             with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
# #                 sf.write(
# #                     f.name,
# #                     audio_np = np.concatenate([
# #                         np.mean(f.to_ndarray(), axis=0) for f in frames
# #                     ])

# #                     # np.concatenate([f.to_ndarray()[0] for f in frames]),
# #                     samplerate=frames[0].sample_rate
# #                 )
# #                 audio, sr = librosa.load(f.name, sr=None)
# #                 label = predict(audio, sr)
# #                 st.success(f"🧠 Predicted Sentiment: **{label}**")
# #     else:
# #         st.info("🎙️ Click START to record and STOP before analyzing.")
# if ctx.audio_receiver and ctx.audio_processor:
#     frames = ctx.audio_processor.recorded_frames
#     st.info(f"🎧 Recorded frames: {len(frames)}")

#     if len(frames) > 20:
#         st.success("✅ Recording complete. Click below to analyze.")

#         if st.button("🔍 Analyze Recorded Audio"):
#             with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
#                 # Convert to mono and flatten
#                 audio_np = np.concatenate([
#                     np.mean(frame.to_ndarray(), axis=0) for frame in frames
#                 ])
#                 sf.write(f.name, audio_np, samplerate=frames[0].sample_rate)

#                 # Load & predict
#                 audio, sr = librosa.load(f.name, sr=None)
#                 label = predict(audio, sr)
#                 st.success(f"🧠 Predicted Sentiment: **{label}**")
#     else:
#         st.warning("⚠️ Please record at least 2-3 seconds of audio before analyzing.")

# ----------------------------------

# import streamlit as st
# import librosa
# import numpy as np
# import tensorflow as tf
# import pickle
# import tempfile
# import soundfile as sf
# import av

# from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

# # Load model & label encoder
# model = tf.keras.models.load_model("sentiment_cnn_model.h5")
# le = pickle.load(open("le.pkl", "rb"))

# st.set_page_config(page_title="🎧 Audio Sentiment Analyzer", layout="centered")

# st.title("🎤 Audio Sentiment Detection using CNN")

# # --------------------- Feature Extraction ---------------------
# def extract_features(audio, sr):
#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
#     return np.mean(mfcc.T, axis=0)

# def predict(audio, sr):
#     feature = extract_features(audio, sr)
#     feature = feature.reshape(1, 40, 1, 1)
#     prediction = model.predict(feature)
#     label = le.inverse_transform([np.argmax(prediction)])[0]
#     return label

# # --------------------- Upload Section ---------------------
# st.header("📤 Upload Audio File")
# audio_file = st.file_uploader("Choose a file (WAV, MP3, M4A)", type=["wav", "mp3", "m4a"])

# if audio_file is not None:
#     st.audio(audio_file)
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#         tmp.write(audio_file.read())
#         tmp_path = tmp.name

#     audio, sr = librosa.load(tmp_path, sr=None)
#     label = predict(audio, sr)
#     st.success(f"🧠 Predicted Sentiment: **{label}**")

# # --------------------- Record Section ---------------------
# st.header("🎙️ Or Record Your Voice")

# class AudioProcessor(AudioProcessorBase):
#     def __init__(self):
#         self.frames = []

#     def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
#         self.frames.append(frame)
#         return frame

# ctx = webrtc_streamer(
#     key="speech",
#     mode=WebRtcMode.SENDRECV,
#     audio_processor_factory=AudioProcessor,
#     media_stream_constraints={"audio": True, "video": False},
# )

# if ctx.audio_processor:
#     frames = ctx.audio_processor.frames
#     st.info(f"🎧 Frames recorded: {len(frames)}")

#     if len(frames) > 30:
#         st.success("✅ Recording complete.")
#         if st.button("🔍 Analyze Recorded Audio"):
#             try:
#                 with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
#                     # Convert all frames to mono and flatten
#                     audio_np = np.concatenate([
#                         np.mean(frame.to_ndarray(), axis=0) for frame in frames
#                     ])
#                     sf.write(f.name, audio_np, samplerate=frames[0].sample_rate)

#                     # Predict sentiment
#                     audio, sr = librosa.load(f.name, sr=None)
#                     label = predict(audio, sr)
#                     st.success(f"🧠 Predicted Sentiment: **{label}**")
#             except Exception as e:
#                 st.error(f"❌ Error: {e}")
#     else:
#         st.warning("⚠️ Please record at least 2-3 seconds of speech.")
# ---------------------------

# import streamlit as st
# import librosa
# import numpy as np
# import tensorflow as tf
# import pickle
# import tempfile
# import soundfile as sf
# import av

# from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

# # Load model and label encoder
# model = tf.keras.models.load_model("sentiment_cnn_model.h5")
# le = pickle.load(open("le.pkl", "rb"))

# st.set_page_config(page_title="🎧 Audio Sentiment Analyzer", layout="centered")
# st.title("🎤 Audio Sentiment Detection using CNN")

# # ----- Feature extraction and prediction -----
# def extract_features(audio, sr):
#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
#     return np.mean(mfcc.T, axis=0)

# def predict(audio, sr):
#     feature = extract_features(audio, sr)
#     feature = feature.reshape(1, 40, 1, 1)
#     prediction = model.predict(feature)
#     label = le.inverse_transform([np.argmax(prediction)])[0]
#     return label

# # ----- Upload Section -----
# st.header("📤 Upload Audio File")
# audio_file = st.file_uploader("Choose a file (WAV, MP3, M4A)", type=["wav", "mp3", "m4a"])

# if audio_file is not None:
#     st.audio(audio_file)
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#         tmp.write(audio_file.read())
#         tmp_path = tmp.name

#     audio, sr = librosa.load(tmp_path, sr=None)
#     label = predict(audio, sr)
#     st.success(f"🧠 Predicted Sentiment: **{label}**")

# # ----- Record Section -----
# st.header("🎙️ Or Record Your Voice")

# class AudioProcessor(AudioProcessorBase):
#     def __init__(self):
#         self.frames = []
#         self.predicted = False
#         self.result_label = ""

#     def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
#         self.frames.append(frame)

#         # Predict automatically when enough frames are collected
#         if not self.predicted and len(self.frames) >= 60:
#             try:
#                 audio_np = np.concatenate([
#                     np.mean(f.to_ndarray(), axis=0) for f in self.frames
#                 ])
#                 with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
#                     sf.write(f_out.name, audio_np, samplerate=frame.sample_rate)
#                     audio, sr = librosa.load(f_out.name, sr=None)
#                     self.result_label = predict(audio, sr)
#                     self.predicted = True
#             except Exception as e:
#                 self.result_label = f"Error: {e}"
#                 self.predicted = True

#         return frame

# ctx = webrtc_streamer(
#     key="speech",
#     mode=WebRtcMode.SENDRECV,
#     audio_processor_factory=AudioProcessor,
#     media_stream_constraints={"audio": True, "video": False},
# )

# # Show prediction automatically after recording
# if ctx.audio_processor:
#     frames = ctx.audio_processor.frames
#     if len(frames) > 0:
#         st.info(f"🎧 Frames recorded: {len(frames)}")

#     if ctx.audio_processor.predicted:
#         label = ctx.audio_processor.result_label
#         if "Error" in label:
#             st.error(f"❌ {label}")
#         else:
#             st.success(f"🧠 Predicted Sentiment: **{label}**")
# -------------------------------------------

import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import soundfile as sf
import tempfile
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av

# Load the trained CNN model and LabelEncoder
model = tf.keras.models.load_model("sentiment_cnn_model.h5")
le = pickle.load(open("le.pkl", "rb"))

# Function to extract MFCC features
def extract_features(file_path):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0).reshape(1, 40, 1, 1)

# Predict sentiment from audio
def predict_sentiment(file_path):
    features = extract_features(file_path)
    prediction = model.predict(features)
    label = le.inverse_transform([np.argmax(prediction)])[0]
    confidence = np.max(prediction)
    return label, confidence

# Streamlit UI Setup
st.set_page_config(page_title="🎵 Audio Sentiment Classifier", layout="centered")
st.title("🎵 Audio Sentiment Analysis (CNN + Streamlit)")

# ==== Audio File Upload Section ====
st.header("📤 Upload Audio")
uploaded_file = st.file_uploader("Upload a WAV/MP3 file", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    label, confidence = predict_sentiment(tmp_path)
    st.success(f"**Predicted Sentiment:** {label}")
    st.write(f"Confidence: {confidence*100:.2f}%")

# ==== Record Audio via Microphone ====
st.header("🎙 Record Your Voice")

class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.recorded_frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.recorded_frames.append(frame)
        return frame

    def get_wav_data(self):
        if not self.recorded_frames:
            return None
        # Properly extract raw PCM data as numpy array
        audio_array = np.concatenate([f.to_ndarray().flatten() for f in self.recorded_frames])
        return audio_array, self.recorded_frames[0].sample_rate

webrtc_ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

if webrtc_ctx and webrtc_ctx.state.playing:
    if st.button("🛑 Stop & Analyze"):
        audio_processor = webrtc_ctx.audio_processor
        if audio_processor is not None:
            audio_data = audio_processor.get_wav_data()
            if audio_data:
                wav_data, sr = audio_data
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    sf.write(f.name, wav_data, sr)
                    label, confidence = predict_sentiment(f.name)
                    st.success(f"**Predicted Sentiment (Recorded):** {label}")
                    st.write(f"Confidence: {confidence*100:.2f}%")
            else:
                st.warning("No audio recorded yet. Please try again.")
