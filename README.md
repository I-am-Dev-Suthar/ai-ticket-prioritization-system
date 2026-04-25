<img width="1167" height="314" alt="Screenshot 2026-04-25 at 7 46 10 PM" src="https://github.com/user-attachments/assets/6534ffd5-c77c-4a12-ace6-10f06ec5e898" /># AI Ticket Prioritization System

## Problem

Support teams receive large volumes of tickets.  
Manually prioritizing them is slow, inconsistent, and error-prone.

---

## Solution

This system uses a hybrid approach:

- Machine Learning (TF-IDF + Logistic Regression)
- Rule-based overrides for critical scenarios

It automatically assigns:
- Priority (high / medium / low)
- Confidence score
- Review flag
- Responsible team

---

## Features

- ML-based ticket classification
- Rule-based priority override
- Confidence-based review system
- REST API using FastAPI
- SQLite database integration
- Evaluation with real-world test cases

---

## API Endpoints

- POST /tickets → Create ticket
- GET /tickets → View all tickets
- GET /tickets/{id} → Get ticket by ID
- PUT /tickets/{id} → Update ticket
- DELETE /tickets/{id} → Delete ticket
- GET /metrics → System performance metrics

---

## Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Access API docs at:
http://127.0.0.1:8000/docs

## Example

Input:
"payment system completely down"

Output:
- priority: high  
- confidence: ~0.90  
- team: Critical Response Team

---

## Tech Stack

- Python
- FastAPI
- Scikit-learn
- SQLite
