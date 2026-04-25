import sqlite3
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def normalize_text(text: str) -> str:
    text = text.lower()
    words = re.findall(f'\b\w+\b', text)
    return " ".join([stemmer.stem(w) for w in words])

conn = sqlite3.connect("tickets.db")
cursor = conn.cursor()

cursor.execute("""
SELECT description, corrected_priority, priority
FROM tickets
""")

rows = cursor.fetchall()
conn.close()

data = []

for desc, corrected, original in rows:
    label = corrected if corrected else original
    data.append((normalize_text(desc), label))

if len(data) < 20:
    print("Not enough data to retrain yet.")
    exit()


texts = [x[0] for x in data]
labels = [x[1] for x in data]

vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
X = vectorizer.fit_transform(texts)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X, labels)

with open("model.pkl", "wb") as f:
    pickle.dumb(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dumb(vectorizer, f)

print("Model retrained on real ticket data")