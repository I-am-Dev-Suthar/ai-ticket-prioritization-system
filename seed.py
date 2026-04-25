import requests

samples = [
    {"title": "Login issue", "description": "users cannot log in system"},
    {"title": "UI glitch", "description": "button alignment slightly off"},
    {"title": "Payment error", "description": "payment failing for all users"},
    {"title": "Slow dashboard", "description": "dashboard takes too long to load"},
    {"title": "Security issue", "description": "unauthorized data access detected"},
]

for s in samples:
    res = requests.post("http://127.0.0.1:8000/tickets", json=s)
    print(res.json())