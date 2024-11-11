def extract_features(file_path):
    """Extract audio features from the given file path"""
    y, sr = librosa.load(file_path)
    
    # Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    
    # Concatenate features
    features = np.concatenate([
        np.mean(mfccs, axis=1),
        [np.mean(spectral_centroid)],
        [np.mean(spectral_bandwidth)],
        [np.mean(spectral_rolloff)],
        [np.mean(zero_crossing_rate)]
    ])
    
    return features

import numpy as np
import tensorflow as tf
import librosa

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Load and preprocess the audio file
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)  # Load audio with original sampling rate
    return audio, sr

# Placeholder function for feature extraction - replace with actual function
def extract_features(audio, sr):
    # Replace this function with the actual feature extraction function.
    # Example (using MFCCs as placeholder):
    features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(features.T, axis=0)

# Load audio
audio_file = '3.wav'
audio, sr = load_audio(audio_file)

# Extract features
features = extract_features(audio, sr)
features = np.expand_dims(features, axis=0)  # Reshape for model input if needed

# Predict class
prediction = model.predict(features)
predicted_class = np.argmax(prediction, axis=1)

print("Predicted Class:", predicted_class)
