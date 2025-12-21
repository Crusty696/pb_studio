
import sqlite3
from pathlib import Path

def check_version():
    db_path = Path("data/project.db")
    if not db_path.exists():
        print("Database not found!")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT * FROM schema_info")
        rows = cursor.fetchall()
        print("Schema Info Table Content:")
        for row in rows:
            print(row)
    except Exception as e:
        print(f"Error reading schema_info: {e}")
        
    conn.close()

if __name__ == "__main__":
    check_version()
