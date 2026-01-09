from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle

app = FastAPI()

# ---- CORS FIX ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
      "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- LOAD MODEL ----
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

class Message(BaseModel):
    text: str

@app.post("/predict")
def predict_spam(data: Message):
    text_vec = vectorizer.transform([data.text])
    proba = model.predict_proba(text_vec)[0][1]  # probability of spam
    prediction = model.predict(text_vec)[0]

    confidence = round(proba * 100, 2)
    message = data.text.lower()

    # ---- CATEGORY DETECTION (rule-based for now) ----
    category = "Personal / Informational"

    if any(word in message for word in ["free", "win", "offer", "discount", "sale", "deal"]):
        category = "Promotional / Ads"

    if any(word in message for word in ["loan", "credit", "invest", "upi", "bitcoin"]):
        category = "Financial / Loan Scam"

    if any(word in message for word in ["package", "delivery", "customs", "shipment"]):
        category = "Delivery Scam / Logistics"

    if any(word in message for word in ["account", "verify", "password", "otp", "kyc"]):
        category = "Banking / Verification Scam"

    # ---- RISK & EMOJI LOGIC ----
    if confidence < 40:
        risk = "Low"
        emoji = "😊"
    elif confidence < 65:
        risk = "Medium"
        emoji = "😐"
    elif confidence < 85:
        risk = "High"
        emoji = "😟"
    else:
        risk = "Very High"
        emoji = "😱"

    # ---- FINAL LABEL ----
    prediction_label = "SPAM" if prediction == 1 else "SAFE"

    return {
        "prediction": prediction_label,
        "confidence": confidence,
        "risk": risk,
        "emoji": emoji,
        "category": category
    }
