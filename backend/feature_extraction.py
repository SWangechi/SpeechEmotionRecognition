import numpy as np
import librosa

def extract_features(data, sample_rate):
    result = np.array([])

    # 1️⃣ Zero Crossing Rate (1 feature)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # 1 feature

    # 2️⃣ Chroma STFT (12 features)
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # +12 features

    # 3️⃣ MFCCs (40 features)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfcc))  # +40 features

    # 4️⃣ Root Mean Square Energy (1 feature)
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # +1 feature

    # 5️⃣ Mel Spectrogram (108 features)
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate, n_mels=108).T, axis=0)
    result = np.hstack((result, mel))  # +108 features

    # 🔥 Check the final feature length
    if result.shape[0] != 162:
        print(f"❌ Feature shape mismatch: Expected 162, got {result.shape[0]}")
    else:
        print(f"✅ Extracted Feature Shape: {result.shape[0]}")

    return result

# ✅ TEST FEATURE EXTRACTION
if __name__ == "__main__":
    file_path = "C:/Users/User/Documents/Speech_Emotion_Recognition/crema/AudioWAV/1001_DFA_FEA_XX.wav"
    
    try:
        data, sample_rate = librosa.load(file_path, sr=22050)
        features = extract_features(data, sample_rate)
        print(f"✅ Final extracted feature shape: {features.shape[0]}")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
