import sqlite3
import os

db_path = "test_hybrid.db"
# We need to re-run the part of validate_hybrid.py that populates the DB if it doesn't exist
# Or just run this after validate_hybrid.py fails (it leaves the DB)

if os.path.exists(db_path):
    db = sqlite3.connect(db_path)
    print("--- FTS Content ---")
    rows = db.execute("SELECT * FROM entries_fts").fetchall()
    for r in rows:
        print(r)
        
    print("\n--- Testing MATCH '알고리즘' ---")
    res = db.execute("SELECT id, text, bm25(entries_fts) FROM entries_fts WHERE text MATCH '알고리즘'").fetchall()
    print(f"Results for '알고리즘': {res}")
    
    print("\n--- Testing MATCH 'task' ---")
    res = db.execute("SELECT id, text, bm25(entries_fts) FROM entries_fts WHERE text MATCH 'task'").fetchall()
    print(f"Results for 'task': {res}")
else:
    print("DB not found")
