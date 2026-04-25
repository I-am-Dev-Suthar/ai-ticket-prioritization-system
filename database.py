import sqlite3

def get_connection():
    return sqlite3.connect("tickets.db")

def create_table():

    conn = sqlite3.connect("tickets.db")
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tickets (
        id INTEGER PRIMARY KEY,
        title TEXT,
        description TEXT,
        priority TEXT,
        confidence REAL,
        used_rule INTEGER,
        needs_review INTEGER
    )
    """)
    
    conn.commit()
    conn.close()