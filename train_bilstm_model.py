import pandas as pd
import re
import nltk
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

df = pd.read_csv("depression.csv")
df.dropna(inplace=True)

# Balance data
df_0 = df[df['label'] == 0]
df_1 = df[df['label'] == 1]
df_balanced = pd.concat([df_0.sample(len(df_1)), df_1])

# Clean text
stop_words = set(stopwords.words("english"))
lm = WordNetLemmatizer()
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    return ' '.join([lm.lemmatize(word) for word in text if word not in stop_words])

df_balanced['cleaned'] = df_balanced['text'].apply(clean_text)

# Tokenizer
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df_balanced['cleaned'])
X = tokenizer.texts_to_sequences(df_balanced['cleaned'])
X = pad_sequences(X, maxlen=100)
y = df_balanced['label'].values

# Train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# BiLSTM model
model = Sequential()
model.add(Embedding(10000, 128, input_length=100))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)

# Save model + tokenizer
model.save("bilstm_model.h5")
with open("tokenizer.pickle", "wb") as f:
    pickle.dump(tokenizer, f)
