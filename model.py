from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from nltk.stem import PorterStemmer
import re

stemmer = PorterStemmer()

# =========================
# TEXT CLEANING
# =========================
def normalize_text(text: str) -> str:
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)

    stemmed_words = []
    for word in words:
        stemmed_words.append(stemmer.stem(word))

    return " ".join(stemmed_words)

# =========================
# DATASET
# =========================
data = [

# =========================
# HIGH (clear blocking + system-wide + security + data loss)
# =========================
("file uploads failing for all users continuously", "high"),
("server overloaded causing system-wide failure", "high"),
("backup system failing completely every time", "high"),
("authentication failing for all users", "high"),
("requests failing across entire system", "high"),
("critical service unavailable globally", "high"),
("system completely down for all users", "high"),
("authentication service blocking all access", "high"),
("payment system completely unavailable", "high"),
("database connection failed system-wide", "high"),
("data lost after production update", "high"),
("security breach detected in production", "high"),
("api service completely not responding", "high"),
("users cannot log into system at all", "high"),
("entire application is down", "high"),
("critical crash on startup for all users", "high"),
("core service failure affecting all traffic", "high"),
("transaction system completely broken", "high"),
("backend infrastructure not responding", "high"),
("system outage affecting entire platform", "high"),
("all requests failing in production", "high"),
("authentication completely broken", "high"),
("live service unavailable globally", "high"),
("data corruption in main database", "high"),
("payment gateway fully down", "high"),
("system crash affecting all operations", "high"),
("immediate fix required system down", "high"),
("urgent login failure blocking all users", "high"),
("critical system failure after deployment", "high"),
("entire platform inaccessible", "high"),
("security vulnerability exposing user data", "high"),
("password reset completely broken", "high"),
("user authentication failing", "high"),
("core feature not working", "high"),
("login page not rendering", "high"),
("backup system completely failing", "high"),
("file uploads failing for all users", "high"),
("memory leak crashing system", "high"),
("payment service timing out repeatedly", "high"),
("file uploads failing consistently for users", "high"),
("backup process failing every time", "high"),
("account locked for all valid users", "high"),
("memory leak degrading system continuously", "high"),
("unauthorized data access detected in system", "high"),
("server overloaded causing request failures", "high"),
("payment system declining transactions repeatedly", "high"),

# =========================
# MEDIUM (partial failure / intermittent / degraded)
# =========================
("file uploads failing intermittently but retry works", "medium"),
("server overloaded during peak but recovers", "medium"),
("backup process failing occasionally", "medium"),
("authentication failing for some users only", "medium"),
("requests failing under load but not always", "medium"),
("login failing intermittently for some users", "medium"),
("payment processing occasionally failing", "medium"),
("api responses slow under heavy load", "medium"),
("search returning incomplete results", "medium"),
("dashboard loading slowly sometimes", "medium"),
("file upload failing for large files", "medium"),
("email notifications delayed occasionally", "medium"),
("data sync inconsistent across devices", "medium"),
("checkout process failing on some attempts", "medium"),
("reports not generating correctly sometimes", "medium"),
("session timeout happening too early", "medium"),
("filters not working properly at times", "medium"),
("form submission failing intermittently", "medium"),
("background jobs running slower than expected", "medium"),
("profile updates not saving immediately", "medium"),
("notification delivery inconsistent", "medium"),
("system performance degraded under load", "medium"),
("payment validation failing intermittently", "medium"),
("api timeout during peak traffic", "medium"),
("search feature sometimes returns errors", "medium"),
("upload process unstable for large files", "medium"),
("data update delays in dashboard", "medium"),
("login works but occasionally fails", "medium"),
("checkout calculation sometimes incorrect", "medium"),
("service response slower than usual", "medium"),
("some users experiencing login issues", "medium"),
("intermittent errors in api calls", "medium"),
("partial failure in data processing", "medium"),
("notifications sometimes not delivered", "medium"),
("system slower during peak hours", "medium"),
("backup process failing occasionally", "medium"),
("api errors under load", "medium"),
("file upload failing sometimes", "medium"),
("data sync breaking intermittently", "medium"),
("memory leak causing slow performance", "medium"),
("pagination not working correctly", "medium"),
("dashboard showing incorrect values", "medium"),
("order history missing entries", "medium"),
("reports showing wrong data", "medium"),
("password email delayed", "medium"),
("password reset email not received sometimes", "medium"),
("file upload failing sometimes but retry works", "medium"),
("payment timeout occasionally", "medium"),
("login fails intermittently", "medium"),
("session expiring slightly early", "medium"),
("user sessions expiring quickly but usable", "medium"),
("navigation feature not functioning correctly", "medium"),
("form submission sometimes fails", "medium"),

# =========================
# LOW (cosmetic / UI / non-blocking issues)
# =========================
("button not working visually but functional", "low"),
("link broken in footer only", "low"),
("icons missing but system works", "low"),
("tooltip not working but no impact", "low"),
("minor ui glitch not affecting usage", "low"),
("button alignment slightly off", "low"),
("tooltip not appearing sometimes", "low"),
("hover effect not smooth", "low"),
("minor spacing issue in layout", "low"),
("icon misaligned in header", "low"),
("text overlapping on small screens", "low"),
("color mismatch in theme styling", "low"),
("font rendering slightly inconsistent", "low"),
("border spacing uneven in card", "low"),
("animation not smooth in ui", "low"),
("checkbox alignment slightly off", "low"),
("label spacing not perfect", "low"),
("minor css issue in layout", "low"),
("design inconsistency in components", "low"),
("hover state visually incorrect", "low"),
("padding inconsistent across pages", "low"),
("small visual glitch in dashboard", "low"),
("ui layout slightly misaligned", "low"),
("button style not matching design", "low"),
("visual polish needs improvement", "low"),
("checkbox not responding visually", "low"),
("button not working visually", "low"),
("tooltip not showing", "low"),
("icon missing but functionality works", "low"),
("typo in faq section", "low"),
("minor flicker on refresh only visual", "low"),
("visual glitch not affecting usage", "low"),
("broken link in footer only", "low"),

# =========================
# NEUTRAL (system OK)
# =========================
("system working normally", "low"),
("no issues detected in system", "low"),
("everything functioning as expected", "low"),
("application running smoothly", "low"),
("all services operational", "low"),
("no errors observed currently", "low"),
("system stable and healthy", "low"),
("no anomalies found in logs", "low"),
("platform operating normally", "low"),
("all features working fine", "low"),

# =========================
# NEGATIVE / EDGE CASES
# =========================
("issue is not critical", "low"),
("this is not urgent", "low"),
("not a major problem", "low"),
("not affecting core functionality", "low"),
("not blocking any users", "low"),
("system not critical right now", "low"),
("no urgent action required", "low"),
("issue exists but low priority", "low"),
("minor issue not impacting workflow", "low"),
("not a serious concern", "low"),
("problem is small and manageable", "low"),
("not impacting production systems", "low"),
("can be fixed later not urgent", "low"),
("this is not blocking anything", "low"),
("low impact issue observed", "low"),

("not urgent but needs fixing", "medium"),
("working fine but slightly slow", "medium"),
("system is usable but has issues", "medium"),
("not critical but noticeable bug", "medium"),
("issue occurs sometimes not always", "medium"),
("not blocking but affecting experience", "medium"),
("system works but not smoothly", "medium"),
("delays observed but functionality works", "medium"),
("not urgent however should be checked", "medium"),
("problem exists but not severe", "medium"),
("minor disruption in functionality", "medium"),
("occasionally failing but retry works", "medium"),
("feature works inconsistently", "medium"),
("not critical but affects some users", "medium"),
("partial issue but system still usable", "medium"),
]

# =========================
# PREPARE DATA
# =========================
texts = [normalize_text(t[0]) for t in data]
labels = [t[1] for t in data]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# =========================
# VECTORIZATION
# =========================
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english")

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# MODEL TRAINING
# =========================
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)

# =========================
# EVALUATION
# =========================
preds = model.predict(X_test_vec)

print("\n--- Classification Report ---")
print(classification_report(y_test, preds))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, preds))

# =========================
# SAVE MODEL
# =========================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel saved successfully.")