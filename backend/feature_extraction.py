import librosa
import numpy as np

def extract_features(data, sr):
    try:
        result = np.array([])

        # Zero Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        result = np.hstack((result, zcr))

        # Chroma Feature
        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        result = np.hstack((result, chroma_stft))

        # MFCC
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13).T, axis=0)
        result = np.hstack((result, mfcc))

        # Root Mean Square Energy
        rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        result = np.hstack((result, rms))

        # Mel Spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
        result = np.hstack((result, mel))

        return result

    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None  
