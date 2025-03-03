from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import librosa
import numpy as np
import pickle
import os
from backend.feature_extraction import extract_features

app = FastAPI(title="VibeCheckAI - Audio Emotion Recognition")

@app.get("/")
def home():
    return {"message": "VibeCheckAI is running!"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct paths relative to backend/
MODEL_PATH = os.path.join(BASE_DIR, "xgboost_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

# Load the model
try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ ERROR LOADING MODEL: {str(e)}")

# Load the label encoder
try:
    with open(ENCODER_PATH, "rb") as encoder_file:
        label_encoder = pickle.load(encoder_file)
    print("✅ Label Encoder loaded successfully!")
except Exception as e:
    print(f"❌ ERROR LOADING LABEL ENCODER: {str(e)}")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/predict/")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        if file is None:
            raise HTTPException(status_code=400, detail="No file uploaded")

        print("📂 Received file:", file.filename)

        # Save the file temporarily
        temp_filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(temp_filepath, "wb") as f:
            f.write(await file.read())

        print(f"📁 File saved at: {temp_filepath}")

        # Load the audio file
        data, sr = librosa.load(temp_filepath, sr=22050)
        print("🎵 Audio loaded - Sample rate:", sr, "Length:", len(data))

        # Extract features
        features = np.array(extract_features(data, sr))
        features = features.reshape(1, -1)  # Ensure 2D shape

        print("📊 Extracted features shape:", features.shape)

        # Predict emotion (numerical output)
        prediction = model.predict(features)[0]

        # Convert prediction to emotion label
        predicted_emotion = label_encoder.inverse_transform([prediction])[0]

        print("🔮 Predicted Emotion:", predicted_emotion)

        return {"emotion": predicted_emotion}

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing the audio file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
