from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required resources (first-time only)
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow requests from Netlify frontend

# Load model and tokenizer
try:
    model = load_model("bilstm_model.h5")
    with open("tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")

# Preprocessing setup
stop_words = set(stopwords.words("english"))
lm = WordNetLemmatizer()

# Text cleaning function
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    return ' '.join([lm.lemmatize(word) for word in words if word not in stop_words])

# Root route for test
@app.route("/", methods=["GET"])
def home():
    return "✅ Backend is up and running!"

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' in request"}), 400

        raw_text = data["text"]
        cleaned = clean_text(raw_text)

        # Keyword-based override (optional shortcut)
        keywords = ["depression", "depressed", "hopeless", "worthless", "anxious", "empty", "suicidal", "down", "lonely"]
        if any(word in cleaned.split() for word in keywords):
            return jsonify({
                "result": "Anxiety/Depression",
                "confidence": 95.0,
                "note": "Detected based on strong keywords"
            })

        # Model prediction
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=100)
        prob = model.predict(padded)[0][0]
        label = "Anxiety/Depression" if prob > 0.5 else "Normal"
        confidence = round(float(prob) * 100, 2)

        return jsonify({"result": label, "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
