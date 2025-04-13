
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
model = pickle.load(open("CV_BestModel.sav", "rb"))
vectorizer = pickle.load(open("vectorizer.sav", "rb"))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    return ' '.join([ps.stem(word) for word in text if word not in stop_words])

@app.route("/", methods=["GET"])
def home():
    return "✅ Anxiety/Depression Detection API is Live!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' in request"}), 400

        raw = data["text"]
        cleaned = clean_text(raw)
        features = vectorizer.transform([cleaned])
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

        return jsonify({"result": "Normal", "confidence": round((1 - prob) * 100, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
