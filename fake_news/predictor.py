import os
import joblib
from sklearn.exceptions import NotFittedError

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")
VEC_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

_model = None
_vectorizer = None


def _load_once():
    global _model, _vectorizer

    if _model is None or _vectorizer is None:
        print("📦 Loading model from:", MODEL_PATH)
        print("📦 Loading vectorizer from:", VEC_PATH)

        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VEC_PATH)


def predict_headline(text: str):
    _load_once()

    # ------------------------
    # BASIC VALIDATION
    # ------------------------
    if not text or not text.strip():
        return {
            "prediction": "INVALID",
            "confidence": 0.0,
            "risk": "Low",
            "reason": "Empty input"
        }

    # ------------------------
    # SHORT TEXT GUARD
    # ------------------------
    if len(text.split()) < 6:
        return {
            "prediction": "UNCERTAIN",
            "confidence": 0.0,
            "risk": "Medium",
            "reason": "Text too short for reliable prediction"
        }

    # ------------------------
    # VECTORIZE
    # ------------------------
    try:
        X = _vectorizer.transform([text])
    except NotFittedError:
        raise RuntimeError("TF-IDF vectorizer is not fitted")

    # ------------------------
    # MODEL DECISION
    # ------------------------
    score = float(_model.decision_function(X)[0])

    # Neutral zone (VERY IMPORTANT)
    if abs(score) < 0.5:
        return {
            "prediction": "UNCERTAIN",
            "confidence": round(abs(score), 2),
            "risk": "Medium",
            "reason": "Low model confidence"
        }

    pred = _model.predict(X)[0]

    return {
        "prediction": "REAL" if pred == 1 else "FAKE",
        "confidence": round(abs(score), 2),
        "risk": "Low" if pred == 1 else "High"
    }