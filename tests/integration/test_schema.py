
import pytest
import sqlite3
from pathlib import Path

@pytest.mark.integration
def test_database_schema_v2_columns():
    """Verify that the V2 database migration was successful and new columns exist."""
    # This test expects the database to be at a specific location.
    # In a clean CI/CD, we would initialize a test DB. 
    # For this audit, we check the 'dev' database if it exists, or skip.
    db_path = Path("data/project.db")
    
    if not db_path.exists():
        pytest.skip("Local 'data/project.db' not found. Cannot verify schema on non-existent DB.")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("PRAGMA table_info(video_clips)")
        columns = {row[1] for row in cursor.fetchall()}
        
        required_columns = {"vision_description", "story_role", "content_fingerprint"}
        missing = required_columns - columns
        
        assert not missing, f"Database table 'video_clips' is missing V2 columns: {missing}"
        
    finally:
        conn.close()
