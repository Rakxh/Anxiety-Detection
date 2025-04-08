from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle, re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Setup
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
CORS(app)

model = load_model("bilstm_model.h5")
tokenizer = pickle.load(open("tokenizer.pickle", "rb"))

stop_words = set(stopwords.words("english"))
lm = WordNetLemmatizer()

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    return ' '.join([lm.lemmatize(word) for word in text if word not in stop_words])

@app.route("/", methods=["GET"])
def home():
    return "✅ Backend is up and running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        raw_text = data["text"]
        cleaned = clean_text(raw_text)

        # Smart keyword logic (temp override)
        positive_keywords = ["happy", "joyful", "grateful", "peaceful", "excited"]
        negative_keywords = ["depressed", "anxious", "suicidal", "hopeless", "worthless", "lonely", "empty"]

        if any(word in cleaned.split() for word in positive_keywords):
            return jsonify({"result": "Normal", "confidence": 95.0, "note": "Positive keyword match"})

        if any(word in cleaned.split() for word in negative_keywords):
            return jsonify({"result": "Anxiety/Depression", "confidence": 95.0, "note": "Negative keyword match"})

        # Model prediction
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=100)
        prob = model.predict(padded)[0][0]
        label = "Anxiety/Depression" if prob > 0.5 else "Normal"
        return jsonify({"result": label, "confidence": round(float(prob)*100, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
