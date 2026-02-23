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
from fake_news.predictor import predict_headline  # ❗UNCHANGED

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
        X = spam_vectorizer.transform([data.text])

        if X.nnz == 0:
            return {
                "prediction": "SAFE",
                "confidence": 0,
                "risk": "Low"
            }

        # Use probability instead of raw decision score
        if hasattr(spam_model, "predict_proba"):
            proba = spam_model.predict_proba(X)[0][1]  # Spam probability

            threshold = 0.7  # You can adjust this (0.6–0.8 recommended)

            if proba >= threshold:
                prediction = "SPAM"
                risk = "High"
            else:
                prediction = "SAFE"
                risk = "Low"

            confidence = round(proba * 100, 2)

        else:
            # Fallback if model doesn't support predict_proba
            pred = spam_model.predict(X)[0]
            prediction = "SPAM" if pred == 1 else "SAFE"
            confidence = 50
            risk = "High" if pred == 1 else "Low"

        return {
            "prediction": prediction,
            "confidence": confidence,
            "risk": risk
        }

    except Exception as e:
        print("❌ SPAM ERROR:", e)
        raise

# ======================
# MALWARE / PHISHING (FIXED)
# ======================
@app.post("/predict-malware")
def predict_malware(data: URLData):
    try:
        raw_data = fetch_url_data(data.url)
        result = malware_risk_model(raw_data)

        prediction = result.get("prediction", "UNKNOWN")

        # 🔥 Convert technical labels to user-friendly labels
        if prediction.lower() == "benign":
            prediction = "SAFE"
        elif prediction.lower() == "malicious":
            prediction = "MALICIOUS"

        return {
            "url": data.url,
            "prediction": prediction,
            "score": result.get("malware_score", 0.0),
            "confidence": result.get("confidence", 0.0)
        }

    except Exception as e:
        print("❌ MALWARE ERROR:", e)
        raise
# ======================
# FAKE NEWS (❗UNCHANGED)
# ======================
@app.post("/predict-fake-news")
def predict_fake_news(data: Headline):
    return predict_headline(data.text)