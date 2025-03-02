import pickle

MODEL_PATH = "C:/Users/User/Documents/Speech_Emotion_Recognition/backend/xgboost_model.pkl"

try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    print("✅ Model loaded successfully!")
    print("🧠 Model type:", type(model))
except Exception as e:
    print(f"❌ ERROR LOADING MODEL: {str(e)}")
