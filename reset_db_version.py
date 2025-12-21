
import sqlite3
from pathlib import Path

def reset_version():
    db_path = Path("data/project.db")
    if not db_path.exists():
        print("Database not found!")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("DROP TABLE IF EXISTS schema_info")
        print("Dropped schema_info table. DB is now effectively V1 (or unversioned).")
        conn.commit()
    except Exception as e:
        print(f"Error dropping schema_info: {e}")
        
    conn.close()

if __name__ == "__main__":
    reset_version()
