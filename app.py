from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask_cors import CORS

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
    return ' '.join([lm.lemmatize(w) for w in text if w not in stop_words])

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    cleaned = clean_text(data['text'])
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100)
    prob = model.predict(padded)[0][0]
    label = "Anxiety/Depression" if prob > 0.5 else "Normal"
    return jsonify({"result": label, "confidence": round(float(prob) * 100, 2)})

if __name__ == "__main__":
    app.run()
