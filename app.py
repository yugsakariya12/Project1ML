from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import traceback
from fastapi import UploadFile, File
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import json
# ======================
# PATHS
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ======================
# IMPORT LOGIC
# ======================


# ======================
# LOAD FAKE NEWS MODEL
# ======================
fake_news_model = joblib.load(os.path.join(MODEL_DIR, "fake_news_model.pkl"))
fake_news_vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

print("✅ Fake News model loaded")

# ======================
# FASTAPI APP
# ======================
app = FastAPI()

# ======================
# GLOBAL ERROR LOGGER (FIXED: only one middleware)
# ======================
@app.middleware("http")
async def log_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        print("\n🔥🔥🔥 SERVER ERROR 🔥🔥🔥")
        print("PATH:", request.url.path)
        traceback.print_exc()
        print("🔥🔥🔥 END ERROR 🔥🔥🔥\n")
        raise

# ======================
# CORS
# ======================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# LOAD SPAM MODELS
# ======================
spam_model = joblib.load(os.path.join(MODEL_DIR, "spam_model.pkl"))
spam_vectorizer = joblib.load(os.path.join(MODEL_DIR, "spam_vectorizer.pkl"))

# LOAD MALWARE MODEL
malware_model = joblib.load(os.path.join(MODEL_DIR, "phishing_rf_model.pkl"))

# ======================
# LOAD IMAGE MODEL (FIXED POSITION)
# NEW (replace with this)
_new_model_path = os.path.join(MODEL_DIR, "phishing_model_final.keras")
_old_model_path = os.path.join(MODEL_DIR, "phishing_model.h5")
_threshold_path = os.path.join(MODEL_DIR, "threshold.json")

if os.path.exists(_new_model_path):
    image_model = tf.keras.models.load_model(_new_model_path)
else:
    image_model = tf.keras.models.load_model(_old_model_path)

if os.path.exists(_threshold_path):
    with open(_threshold_path) as f:
        IMAGE_THRESHOLD = float(json.load(f)["threshold"])
else:
    IMAGE_THRESHOLD = 0.5

print("✅ Image phishing model loaded")
print("✅ Spam model loaded")
print("✅ Spam vectorizer loaded")
print("✅ Image phishing model loaded")

# DEBUG
print("MODEL CLASSES:", spam_model.classes_)
print("NUMBER OF CLASSES:", len(spam_model.classes_))
print(type(fake_news_model))

# ======================
# REQUEST SCHEMAS
# ======================
class Message(BaseModel):
    text: str

class URLData(BaseModel):
    url: str

class Headline(BaseModel):
    text: str

# ======================
# SPAM DETECTION
# ======================
@app.post("/predict")
def predict_spam(data: Message):
    try:
        text = data.text.strip().lower()
        words = text.split()

        if len(words) <= 3:
            return {"prediction": "SAFE", "confidence": 95, "risk": "Low"}

        safe_keywords = [
            "otp","order","delivery","payment","appointment",
            "meeting","train ticket","credited","statement",
            "recharge","transaction","invoice","bill"
        ]

        if any(k in text for k in safe_keywords):
            return {"prediction": "SAFE", "confidence": 90, "risk": "Low"}

        spam_keywords = [
            "lottery","winner","claim","free money","prize",
            "bank verify","account suspended","verify account",
            "click here","urgent action","claim prize","limited offer"
        ]

        if any(k in text for k in spam_keywords):
            return {"prediction": "SPAM", "confidence": 95, "risk": "High"}

        X = spam_vectorizer.transform([text])
        probs = spam_model.predict_proba(X)[0]

        spam_prob = probs[1]

        if spam_prob > 0.80:
            prediction = "SPAM"
            risk = "High"
        elif spam_prob > 0.60:
            prediction = "SPAM"
            risk = "Medium"
        else:
            prediction = "SAFE"
            risk = "Low"

        return {
            "prediction": prediction,
            "confidence": round(spam_prob * 100, 2),
            "risk": risk
        }

    except Exception as e:
        print("SPAM ERROR:", e)
        return {"prediction": "ERROR", "confidence": 0, "risk": "Low"}

# ======================
# MALWARE
# ======================
@app.post("/predict-malware")
def predict_malware(data: URLData):
    try:
        url = data.url.strip().lower()

        if not url:
            return {
                "url": url,
                "prediction": "INVALID",
                "risk_score": 0,
                "confidence": 0,
                "risk_level": "Unknown"
            }

        prediction = malware_model.predict([url])[0]

        try:
            prob = malware_model.predict_proba([url])[0].max()
        except Exception:
            prob = 0.75

        pred_text = str(prediction).lower()

        if pred_text in ["1", "phishing", "malicious", "bad"]:
            result = "MALICIOUS"
            risk = "High"
        else:
            result = "SAFE"
            risk = "Low"

        return {
            "url": url,
            "prediction": result,
            "risk_score": round(prob * 100, 2),
            "confidence": round(prob * 100, 2),
            "risk_level": risk
        }

    except Exception as e:
        print("❌ MALWARE ERROR:", e)
        traceback.print_exc()

        return {
            "url": data.url,
            "prediction": "ERROR",
            "risk_score": 0,
            "confidence": 0,
            "risk_level": "Unknown"
        }

# ======================
# FAKE NEWS
# ======================
@app.post("/predict-fake-news")
def predict_fake_news(data: Headline):
    try:
        text = data.text.strip().lower()

        if not text:
            return {"prediction": "INVALID", "confidence": 0}

        import re
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        X = fake_news_vectorizer.transform([text])
        score = fake_news_model.decision_function(X)[0]

        confidence = min(100, abs(score) * 50)

        if score < -0.6:
            result = "FAKE"
        else:
            result = "REAL"

        return {
            "prediction": result,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        print("❌ FAKE NEWS ERROR:", e)
        traceback.print_exc()

        return {
            "prediction": "ERROR",
            "confidence": 0
        }

# ======================
# IMAGE PHISHING DETECTION
# ======================
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        phishing_prob = float(image_model.predict(img_array, verbose=0)[0][0])

        print(f"RAW: {phishing_prob:.4f} | threshold: {IMAGE_THRESHOLD:.4f}")
        result = "Phishing" if phishing_prob >= IMAGE_THRESHOLD else "Legitimate"
        confidence = phishing_prob if result == "Phishing" else 1.0 - phishing_prob
        return {
              "prediction": result,
               "confidence": round(confidence, 4),
               "phishing_probability": round(phishing_prob, 4),
               }

    except Exception as e:
        print("❌ IMAGE ERROR:", e)
        traceback.print_exc()

        return {
            "prediction": "ERROR",
            "confidence": 0
        }