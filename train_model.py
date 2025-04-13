import pandas as pd
import re
import nltk
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Setup
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    return ' '.join([ps.stem(word) for word in text if word not in stop_words])

# Load data
df = pd.read_csv("depression.csv").dropna()
df["cleaned"] = df["text"].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df["cleaned"]).toarray()
y = df["label"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
model1 = MultinomialNB()
model2 = RandomForestClassifier()
model3 = AdaBoostClassifier()
model4 = SVC(probability=True)

voting_clf = VotingClassifier(estimators=[
    ('nb', model1),
    ('rf', model2),
    ('ada', model3),
    ('svc', model4)
], voting='soft')

# Train
voting_clf.fit(X_train, y_train)

# Evaluate
y_pred = voting_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {acc * 100:.2f}%")

# Save model and vectorizer
pickle.dump(voting_clf, open("CV_BestModel.sav", "wb"))
pickle.dump(vectorizer, open("vectorizer.sav", "wb"))

print("✅ Saved model as CV_BestModel.sav and vectorizer as vectorizer.sav")
