from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import re
import nltk
import gdown
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

app = Flask(__name__)
CORS(app)

# === Google Drive config ===
MODEL_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"  # 🔁 Replace with your actual Drive model file ID
MODEL_PATH = "CV_BestModel.sav"
VECTORIZER_PATH = "vectorizer.sav"

# === Preprocessing ===
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    return ' '.join([ps.stem(word) for word in text if word not in stop_words])

# === Download model if not found ===
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# === Load model and vectorizer ===
model = pickle.load(open(MODEL_PATH, "rb"))
vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))

# === Keyword Triggers (stemmed)
keyword_raw = [
    "stressed", "anxious", "anxiety", "depressed", "depression", "panic", "sad",
    "hopeless", "worthless", "overwhelmed", "numb", "empty", "lonely", "crying", "upset",
    "can't focus", "tired", "burned out", "unmotivated", "no energy", "exhausted",
    "negative thoughts", "losing control", "not good enough", "dark thoughts",
    "self-harm", "cutting", "suicidal", "hate myself", "useless", "burden", "failure",
    "avoiding people", "socially withdrawn", "no one understands", "isolated", "ignored",
    "insomnia", "no sleep", "sleeping all day", "chest pain", "racing heart", "tight chest",
    "shaking", "sweaty", "nausea", "shortness of breath"
]
keyword_triggers = [ps.stem(w) for w in keyword_raw]

# === Negation Triggers ===
negation_patterns = [
    "not happy", "not okay", "not fine", "not good", "not feeling well", "not doing great"
]

# === Routes ===
@app.route("/", methods=["GET"])
def home():
    return "✅ Anxiety/Depression Detection API is Live!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        raw = data["text"]
        cleaned = clean_text(raw)
        cleaned_words = cleaned.split()

        # 🔍 Fallback triggers
        if any(phrase in raw.lower() for phrase in negation_patterns):
            result = 1
            confidence = 91.0
        elif any(kw in cleaned_words for kw in keyword_triggers):
            result = 1
            confidence = 93.0
        else:
            features = vectorizer.transform([cleaned]).toarray()
            result = model.predict(features)[0]
            prob = model.predict_proba(features)[0][1]
            confidence = round(prob * 100, 2)

        if result == 1:
            return jsonify({
                "result": "Anxiety/Depression",
                "confidence": confidence,
                "message": "Please contact your proctor.",
                "mood_support": "Life is very precious. Any problem is solvable. You are not alone.",
                "counsellors": [
                    {"name": "Dr. Jetson Satya Gospal", "phone": "9884078484", "room": "PRP-206"},
                    {"name": "Mr. Felix Emmanuel", "phone": "9442823000", "room": "TT-720"},
                    {"name": "Mr. Clinton Joseph", "phone": "9962325233", "room": "MB-228A"},
                    {"name": "Mr. Parthiban D", "phone": "9443311360", "room": "SJT-326"},
                    {"name": "Mr. R. Muralitharan", "phone": "8981608883", "room": "GDN 151B"}
                ],
                "helpline": "📞 Helpline: 1800-599-0019 (India Mental Health Helpline)",
                "online_support": "🌐 Online Support: https://www.mentalhealthindia.com/",
                "tip": "💡 Tip: Try to identify the root cause of your anxiety and talk to someone you trust."
            })

        # Otherwise → Normal
        if 'prob' in locals():
            normal_conf = round((1 - prob) * 100, 2)
        else:
            normal_conf = 100 - confidence if confidence < 100 else 100.0

        return jsonify({
            "result": "Normal",
            "confidence": normal_conf
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the API
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
