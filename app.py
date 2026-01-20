from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# Malware functions
from malware.analyzer import fetch_url_data, malware_risk_model


app = FastAPI()

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- LOAD SPAM MODEL SAFELY ----
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---- REQUEST MODELS ----
class Message(BaseModel):
    text: str

class URLData(BaseModel):
    url: str


# ======================
#  SPAM DETECTION API
# ======================
@app.post("/predict")
def predict_spam(data: Message):
    text_vec = vectorizer.transform([data.text])
    proba = model.predict_proba(text_vec)[0][1]
    prediction = model.predict(text_vec)[0]

    confidence = round(proba * 100, 2)
    message = data.text.lower()

    # Category
    category = "Personal / Informational"
    if any(x in message for x in ["free", "win", "offer", "discount", "sale", "deal"]):
        category = "Promotional / Ads"
    if any(x in message for x in ["loan", "credit", "invest", "upi", "bitcoin"]):
        category = "Financial / Loan Scam"
    if any(x in message for x in ["package", "delivery", "customs", "shipment"]):
        category = "Delivery Scam / Logistics"
    if any(x in message for x in ["account", "verify", "password", "otp", "kyc"]):
        category = "Banking / Verification Scam"

    # Risk levels
    if confidence < 40:
        risk, emoji = "Low", "😊"
    elif confidence < 65:
        risk, emoji = "Medium", "😐"
    elif confidence < 85:
        risk, emoji = "High", "😟"
    else:
        risk, emoji = "Very High", "😱"

    prediction_label = "SPAM" if prediction == 1 else "SAFE"

    return {
        "prediction": prediction_label,
        "confidence": confidence,
        "risk": risk,
        "emoji": emoji,
        "category": category
    }


# ======================
#  MALWARE / PHISHING API
# ======================
@app.post("/predict-malware")
def predict_malware(data: URLData):
    raw_data = fetch_url_data(data.url)
    result = malware_risk_model(raw_data)

    return {
        "url": data.url,
        "prediction": result["prediction"],
        "score": result["malware_score"],
        "confidence": result["confidence"],
        "raw_data": raw_data
    }
