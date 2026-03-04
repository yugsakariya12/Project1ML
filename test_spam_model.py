import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

spam_model = joblib.load(os.path.join(MODEL_DIR, "spam_model.pkl"))
spam_vectorizer = joblib.load(os.path.join(MODEL_DIR, "spam_vectorizer.pkl"))

test_msgs = [
    "hey meet",
    "call me later",
    "free lottery winner",
    "verify bank account now"
]

X = spam_vectorizer.transform(test_msgs)
probs = spam_model.predict_proba(X)

for m, p in zip(test_msgs, probs):
    print(m, "->", p)