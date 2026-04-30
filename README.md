# 🧠 Multi-Model Machine Learning Security Suite

A comprehensive Machine Learning project that detects and classifies various types of cyber threats and misinformation, including:

- 📧 Spam Detection  
- 🦠 Malware Detection  
- 🎣 Phishing Website Detection  
- 📰 Fake News Detection  

This project demonstrates practical applications of ML in cybersecurity and content filtering.

---

## 🚀 Features

- 🔍 **Spam Detection** – Classifies messages as spam or ham  
- 🦠 **Malware Detection** – Identifies malicious files/software  
- 🎣 **Phishing Detection** – Detects fraudulent websites using URL & content features  
- 📰 **Fake News Detection** – Classifies news as real or fake  

---

## 📁 Project Structure
```
├── fake_news/              # Fake news detection module
├── malware/                # Malware detection module
├── models/                 # Trained ML models
├── app.py                  # Main application (Flask/Streamlit)
├── train_model.py          # Model training script
├── test_spam_model.py      # Spam model testing
├── phishing_data.pkl       # Dataset for phishing detection
├── check_vectorizer.py     # Vectorizer checking
├── fix_vectorizer.py       # Fix vectorizer issues
├── convert_model.py        # Model conversion utilities
├── imageintegrate.py       # Image-based integration (if used)
├── requirements.txt        # Dependencies
├── Procfile                # Deployment config
├── runtime.txt             # Python runtime version
```
---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yugsakariya12/Project1ML.git

# Navigate to project folder
cd your-repo-name

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Usage
python app.py

Then open your browser at:

http://localhost:5000

## 🧠 How It Works
1. Spam Detection
- Text preprocessing (tokenization, stopword removal)
- TF-IDF vectorization
- lassification using ML model

2. Malware Detection
- Feature extraction from files
- Classification using trained model

3. Phishing Detection
- URL-based feature extraction
- ML model predicts legitimacy

4. Fake News Detection
- NLP processing of news content
- Classification into real/fake
