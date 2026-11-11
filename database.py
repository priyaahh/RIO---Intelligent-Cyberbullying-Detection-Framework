
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent / "rio_database.db"


def create_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            comment TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def save_analysis(comment, prediction, confidence):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO analysis_history
        (comment, prediction, confidence, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        comment,
        prediction,
        confidence,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def get_analysis_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM analysis_history
        ORDER BY id DESC
    """)

    records = cursor.fetchall()
    conn.close()

    return records
