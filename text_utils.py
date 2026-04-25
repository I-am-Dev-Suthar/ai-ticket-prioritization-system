import re
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def normalize_text(text: str) -> str:
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)

    stemmed_words = []
    for word in words:
        stemmed_words.append(stemmer.stem(word))

    return " ".join(stemmed_words)