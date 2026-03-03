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
        text = data.text.strip()

        # Very short messages should be SAFE
        if len(text.split()) <= 2:
            return {
                "prediction": "SAFE",
                "confidence": 100,
                "risk": "Low"
            }

        X = spam_vectorizer.transform([text])
        probabilities = spam_model.predict_proba(X)[0]

        # Since model classes are [0,1]
        safe_prob = probabilities[0]   # class 0 = ham
        spam_prob = probabilities[1]   # class 1 = spam

        if spam_prob > safe_prob:
            prediction = "SPAM"
            confidence = round(spam_prob * 100, 2)
            risk = "High" if spam_prob > 0.75 else "Medium"
        else:
            prediction = "SAFE"
            confidence = round(safe_prob * 100, 2)
            risk = "Low"

        return {
            "prediction": prediction,
            "confidence": confidence,
            "risk": risk
        }

    except Exception as e:
        print("❌ SPAM ERROR:", e)
        return {
            "prediction": "ERROR",
            "confidence": 0,
            "risk": "Low"
        }
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
        text = data.text.strip()

        if not text:
            return {
                "prediction": "INVALID",
                "confidence": 0
            }

        X = fake_news_vectorizer.transform([text])

        prediction = fake_news_model.predict(X)[0]
        score = fake_news_model.decision_function(X)[0]

        print("RAW SCORE:", score)

        # Convert margin to smoother confidence
        confidence = round((abs(score) / 2) * 100, 2)

        # Cap confidence at 100
        if confidence > 100:
            confidence = 100

        # 🔥 Only uncertain if score extremely close to zero
        if abs(score) < 0.1:
            result = "UNCERTAIN"
        else:
            result = "REAL" if prediction == 1 else "FAKE"

        return {
            "prediction": result,
            "confidence": confidence
        }

    except Exception as e:
        print("❌ FAKE NEWS ERROR:", e)
        return {
            "prediction": "ERROR",
            "confidence": 0
        }