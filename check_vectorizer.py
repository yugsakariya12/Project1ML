import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VEC_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

vec = joblib.load(VEC_PATH)

print("TYPE:", type(vec))
print("HAS idf_:", hasattr(vec, "idf_"))
print("DICT KEYS:", vec.__dict__.keys())