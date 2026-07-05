import sqlite3

conn = sqlite3.connect("startup.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age_first_funding REAL,
    age_last_funding REAL,
    age_first_milestone REAL,
    age_last_milestone REAL,
    relationships INTEGER,
    funding_rounds INTEGER,
    funding_total_usd REAL,
    milestones INTEGER,
    prediction TEXT
)
""")

conn.commit()
conn.close()

print("Database Created Successfully!")