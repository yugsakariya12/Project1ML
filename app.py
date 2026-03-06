from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import traceback

# ======================
# PATHS
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ======================
# IMPORT LOGIC
# ======================
from malware.analyzer import fetch_url_data, malware_risk_model
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

from fastapi import Request
import traceback

@app.middleware("http")
async def log_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        print("\n🔥🔥🔥 SERVER ERROR 🔥🔥🔥")
        print("PATH:", request.url.path)
        traceback.print_exc()
        print("🔥🔥🔥 END ERROR 🔥🔥🔥\n")
        raise e
# ======================
# GLOBAL ERROR LOGGER (MANDATORY)
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

print("✅ Spam model loaded")
print("✅ Spam vectorizer loaded")

# 🔥 ADD THIS HERE
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
# SPAM DETECTION (FIXED)
# ======================
# ======================
# SPAM DETECTION (PROPER THRESHOLD FIX)
# ======================
@app.post("/predict")
def predict_spam(data: Message):
    try:
        text = data.text.strip().lower()
        words = text.split()

        # Short casual messages
        if len(words) <= 3:
            return {"prediction": "SAFE", "confidence": 95, "risk": "Low"}

        # Legitimate keywords
        safe_keywords = [
            "otp","order","delivery","payment","appointment",
            "meeting","train ticket","credited","statement",
            "recharge","transaction","invoice","bill"
        ]

        if any(k in text for k in safe_keywords):
            return {"prediction": "SAFE", "confidence": 90, "risk": "Low"}

        # Strong spam keywords
        spam_keywords = [
            "lottery","winner","claim","free money","prize",
            "bank verify","account suspended","verify account",
            "click here","urgent action","claim prize","limited offer"
        ]

        if any(k in text for k in spam_keywords):
            return {"prediction": "SPAM", "confidence": 95, "risk": "High"}

        # Model prediction
        X = spam_vectorizer.transform([text])
        probs = spam_model.predict_proba(X)[0]

        safe_prob = probs[0]
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
# =====================
# MALWARE / PHISHING (FIXED)
# ======================
# ======================
# MALWARE / PHISHING (FINAL TUNED VERSION)
# ======================
@app.post("/predict-malware")
def predict_malware(data: URLData):
    try:
        raw_data = fetch_url_data(data.url)
        result = malware_risk_model(raw_data)

        # Convert labels to frontend format
        prediction = result.get("prediction", "Unknown")

        if prediction.lower() == "benign":
            prediction = "SAFE"
        elif prediction.lower() == "malicious":
            prediction = "MALICIOUS"
        elif prediction.lower() == "suspicious":
            prediction = "SUSPICIOUS"

        return {
            "url": data.url,
            "prediction": prediction,
            "risk_score": result.get("malware_score", 0),
            "confidence": result.get("confidence", 0),
            "risk_level": result.get("risk_level", "Unknown")
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
# FAKE NEWS (❗UNCHANGED)
# ======================
@app.post("/predict-fake-news")
def predict_fake_news(data: Headline):
    try:
        text = data.text.strip().lower()

        if not text:
            return {"prediction": "INVALID", "confidence": 0}

        # basic cleaning
        import re
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        X = fake_news_vectorizer.transform([text])

        score = fake_news_model.decision_function(X)[0]

        print("MODEL SCORE:", score)

        # convert to confidence
        confidence = min(100, abs(score) * 50)

        # safer classification logic
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