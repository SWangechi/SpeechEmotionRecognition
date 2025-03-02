from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import librosa
import numpy as np
import pickle
from feature_extraction import extract_features
from pydantic import BaseModel
import io

# Initialize FastAPI
app = FastAPI(title="VibeCheckAI - Audio Emotion Recognition")

# Load the model
MODEL_PATH = "C:/Users/User/Documents/Speech_Emotion_Recognition/backend/xgboost_model.pkl"

try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    print("✅ Model loaded successfully!")
    print("🧠 Model type:", type(model))
except Exception as e:
    print(f"❌ ERROR LOADING MODEL: {str(e)}")

# Define response model
class PredictionResponse(BaseModel):
    emotion: str

@app.post("/predict/", response_model=PredictionResponse)
async def predict_emotion(file: UploadFile = File(...)):
    try:
        # Read and process audio
        audio_bytes = await file.read()
        audio_buffer = io.BytesIO(audio_bytes)  # Convert bytes to file-like object
        data, sr = librosa.load(audio_buffer, sr=22050)  # Load audio with Librosa

        # Extract features
        features = extract_features(data, sr)
        features = np.expand_dims(features, axis=0)  # Ensure correct shape

        # Predict emotion
        prediction = model.predict(features)

        return {"emotion": str(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the audio file: {str(e)}")

# Run FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
