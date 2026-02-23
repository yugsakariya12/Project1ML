import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# LOAD DATA (same data used for training)
df = pd.read_csv("Fake.csv")   # change if needed
texts = df["text"].astype(str)

# LOAD OLD VECTOR (for vocabulary)
old_vec = joblib.load("models/tfidf_vectorizer.pkl")

# RECREATE PROPER VECTOR
new_vec = TfidfVectorizer(
    vocabulary=old_vec.vocabulary_,
    stop_words="english"
)

# FIT IT (THIS CREATES idf_)
new_vec.fit(texts)

# SAVE BACK (overwrite)
joblib.dump(new_vec, "models/tfidf_vectorizer.pkl")

print("✅ Vectorizer FIXED and saved")