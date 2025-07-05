# Audio Sentiment Analysis Using CNN

This project is a Flask-based web application for **audio sentiment analysis**. It allows users to upload audio files, extracts features (MFCCs), and predicts the sentiment (e.g., *positive*, *negative*, *neutral*) using a pre-trained Convolutional Neural Network (CNN) model.

## 📂 Project Structure

```
.
├── inputs audio/                # Sample input audios (if any)
├── templates/                   # HTML templates
│   └── index.html               # Main page template
├── uploads/                     # Folder for uploaded audio files
├── .gitignore                   # Git ignore rules
├── audio-sentiment-analysis-... # Jupyter notebook (for training/demo)
├── le.pkl                       # LabelEncoder pickle file
├── main.py                      # Flask application
├── requirements.txt             # Python dependencies
├── sentiment_cnn_model.h5       # Trained CNN model
```

## 🚀 Features

* Upload audio files (e.g., WAV, MP3)
* Extract MFCC features using **Librosa**
* Predict sentiment using a CNN model
* Display prediction results and provide playback for uploaded audio

## 🛠 Installation

1. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   venv\Scripts\activate    # On Windows
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Running the App

1. **Start the Flask server**

   ```bash
   python main.py
   ```

2. Open your browser and navigate to:

   ```
   http://127.0.0.1:5000/
   ```

3. Upload an audio file and view the predicted sentiment.

## 📦 Requirements

See `requirements.txt` for full details. Key dependencies:

* Flask
* TensorFlow
* Librosa
* scikit-learn

## 📊 Model Details

The core of this project is a **Convolutional Neural Network (CNN)** trained to classify audio clips into sentiment categories. Here’s how the model works:

### 🎛 Feature Extraction

* Audio signals are converted into Mel-Frequency Cepstral Coefficients (MFCCs) using the `librosa` library.
* MFCCs represent the short-term power spectrum of sound, making them ideal for speech and audio processing.
* For each audio clip, 40 MFCC features are extracted and averaged over time to form a compact representation.

### 🧠 CNN Architecture

* The model uses convolutional layers to learn local patterns in the MFCCs.
* Typical architecture:

  * **Conv2D layers** to extract hierarchical feature maps from MFCCs.
  * **MaxPooling layers** to reduce dimensionality and focus on key features.
  * **Dropout layers** to prevent overfitting.
  * **Dense layers** for high-level representation and classification.
* The final output layer uses **Softmax activation** to predict probabilities across sentiment classes.

### 📚 Training Details

* The dataset was labeled with sentiments (e.g., *positive*, *negative*, *neutral*).
* The model was trained using the **categorical cross-entropy loss function** and **Adam optimizer**.
* Early stopping and model checkpointing were used to avoid overfitting and save the best model.

### 🚀 Deployment

* The trained model is saved as `sentiment_cnn_model.h5`.
* A `LabelEncoder` object (`le.pkl`) maps predicted numerical labels back to human-readable sentiment categories.

## 🖥 Example Output

When you upload an audio file, the app will:

1. Process and extract MFCC features from the file.
2. Run the features through the CNN model to predict the sentiment.
3. Display the predicted sentiment on the webpage, such as:

```
🎵 Uploaded Audio: happy_voice_sample.wav
😊 Predicted Sentiment: Positive
```

It also provides an embedded audio player to listen to the uploaded clip.

## 📝 Notes

* Supported audio formats depend on **Librosa** and **audioread**.
* The `uploads/` folder is created automatically if it does not exist.


---

✨ *Built with Flask, TensorFlow, and Librosa for audio sentiment analysis.*
