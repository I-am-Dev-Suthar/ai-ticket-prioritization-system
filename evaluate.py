from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from text_utils import normalize_text

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# =========================
# TEST DATA
# =========================
test_data = [
    ("urgent login failure blocking access", "high"),
    ("small ui misalignment on homepage", "low"),
    ("payment processing error at checkout", "medium"),
    ("entire system crash during operation", "high"),
    ("login issue but not time sensitive", "medium"),
    ("application functioning without issues", "low"),
    ("backend server completely unresponsive", "high"),
    ("minor spelling mistake in header", "low"),
    ("checkout flow taking longer than expected", "medium"),
    ("app freezing immediately after launch", "high"),
    ("email alerts arriving with delay", "medium"),
    ("slight button position inconsistency", "low"),
    ("user data missing after software update", "high"),
    ("unable to upload new profile image", "medium"),
    ("theme colors not matching design", "low"),
    ("password reset feature not working", "high"),
    ("search returning partially incorrect results", "medium"),
    ("text appears too small on screen", "low"),
    ("users being logged out randomly", "high"),
    ("report export feature showing errors", "medium"),
    ("footer layout spacing looks uneven", "low"),
    ("database connection cannot be established", "high"),
    ("images loading slower than usual", "medium"),
    ("hover animation not triggering", "low"),
    ("serious security flaw detected", "high"),
    ("push notifications delayed significantly", "medium"),
    ("header elements not properly aligned", "low"),
    ("application crashing after login attempt", "high"),
    ("shopping cart total calculation incorrect", "medium"),
    ("link styling inconsistent across pages", "low"),
    ("payment service timing out frequently", "high"),
    ("filters not updating search results", "medium"),
    ("icons missing from toolbar", "low"),
    ("unauthorized data access detected", "high"),
    ("dashboard takes too long to load", "medium"),
    ("card components have uneven padding", "low"),
    ("scheduled backup process failing", "high"),
    ("emails rendered with formatting issues", "medium"),
    ("border styling differs between sections", "low"),
    ("api returning internal server errors", "high"),
    ("mobile notifications arriving late", "medium"),
    ("text overlapping on smaller screens", "low"),
    ("complete outage reported by users", "high"),
    ("analytics dashboard showing wrong values", "medium"),
    ("tooltips fail to display on hover", "low"),
    ("file uploads consistently failing", "high"),
    ("user sessions expiring too quickly", "medium"),
    ("checkbox input not responding", "low"),
    ("critical defect found in production", "high"),
    ("pagination control not functioning", "medium"),
    ("login page fails to render", "high"),
    ("faq section contains typo", "low"),
    ("cart updates reflecting slowly", "medium"),
    ("system shutting down unexpectedly", "high"),
    ("reset password email never received", "medium"),
    ("navigation icons slightly off center", "low"),
    ("data synchronization breaking intermittently", "high"),
    ("profile changes not saving immediately", "medium"),
    ("background flicker observed on refresh", "low"),
    ("account locked after valid attempts", "high"),
    ("search suggestions not relevant", "medium"),
    ("broken links present in footer", "low"),
    ("memory leak causing degradation", "high"),
    ("order history missing recent entries", "medium"),
    ("incorrect hover color on buttons", "low"),
    ("payments being declined incorrectly", "high"),
    ("api latency higher than expected", "medium"),
    ("text alignment inconsistent in forms", "low"),
    ("user account disabled without reason", "high"),
    ("reports generation taking excessive time", "medium"),
    ("inconsistent spacing across ui sections", "low"),
    ("server overloaded during peak usage", "high"),
    ("notifications not triggering reliably", "medium"),
    ("images not aligned in gallery view", "low"),
    ("possible security breach identified", "high"),
    ("filters producing inaccurate outputs", "medium"),
    ("dropdown menu not expanding", "low"),
    ("form submission causing application crash", "high"),
    ("exported data missing some fields", "medium"),
    ("minor visual glitch in layout", "low"),
    ("database queries timing out", "high"),
    ("validation rules not applied correctly", "medium"),
    ("text overflowing container bounds", "low"),
    ("fraudulent transaction detected", "high"),
    ("login process slower than normal", "medium"),
    ("checkbox label misaligned", "low"),
    ("backup restoration failing critically", "high"),
    ("email delivery slower than expected", "medium"),
    ("button sizes vary across screens", "low"),
    ("authentication token validation failing", "high"),
    ("dashboard widgets not updating", "medium"),
    ("icons spaced unevenly in menu", "low"),
    ("system failure during upgrade", "high"),
    ("search indexing not updating", "medium"),
    ("tooltip content incorrect", "low"),
    ("file integrity compromised", "high"),
    ("session persistence issues observed", "medium"),
    ("radio button interaction broken", "low"),
]

# =========================
# TEST DATA
# =========================
y_true = []
y_pred = []

wrong_cases = []

for text, actual in test_data:
    normalized = normalize_text(text)
    vec = vectorizer.transform([normalized])
    
    probs = model.predict_proba(vec)[0]
    pred = model.classes_[probs.argmax()]
    confidence = max(probs)

    y_true.append(actual)
    y_pred.append(pred)

    if pred != actual:
        wrong_cases.append((text, actual, pred, round(confidence, 2)))

# =========================
# METRICS
# =========================
print("\n--- Accuracy ---")
print(accuracy_score(y_true, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_true, y_pred))

# =========================
# WRONG CASES ANALYSIS
# =========================
print("\n--- Top Wrong Predictions ---")
for case in wrong_cases[:10]:
    print(case)

# =========================
# REAL WORLD TESTS
# =========================
print("\n--- Real World Tests ---")

real_tests = [
    "app not wrking fix asap",
    "payment failed again bro",
    "ui looks weird on mobile",
    "everything slow today idk why",
    "login broken??",
    "system down again wtf",
    "not urgent just annoying bug",
    "pls fix asap its bad",
]

for t in real_tests:
    vec = vectorizer.transform([normalize_text(t)])
    probs = model.predict_proba(vec)[0]
    pred = model.classes_[probs.argmax()]
    conf = max(probs)

    print(f"{t} -> {pred} ({conf:.2f})")

