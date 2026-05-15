import librosa
import numpy as np

def extract_voice_features(file):
    y, sr = librosa.load(file, sr=22050)

    features = []

    features.append(np.mean(librosa.feature.zero_crossing_rate(y)))
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))

    return features