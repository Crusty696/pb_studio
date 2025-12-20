import os
import shutil
import glob
from pathlib import Path
import logging

def perform_startup_cleanup(root_dir: Path):
    """
    Performs aggressive cleanup of temporary data, caches, and the database 
    to ensure a fresh state on application startup.
    Mimics the behavior of CLEAN_AND_START.bat.
    """
    print("Performing startup cleanup...")
    
    # 1. Database
    # Check both root and data/ subdirectory
    db_locations = [root_dir, root_dir / "data"]
    db_filenames = ["project.db", "project.db-shm", "project.db-wal"]
    
    for loc in db_locations:
        if not loc.exists():
            continue
            
        for db_file in db_filenames:
            db_path = loc / db_file
            if db_path.exists():
                try:
                    os.remove(db_path)
                    print(f"Deleted database: {db_path}")
                except Exception as e:
                    print(f"Failed to delete {db_path}: {e}")

    # 2. Logs (be careful not to delete current log if possible, but here we run before logging)
    log_dir = root_dir / "logs"
    if log_dir.exists():
        try:
            # Delete all .log files
            for log_file in log_dir.glob("*.log"):
                try:
                    os.remove(log_file)
                except Exception as e:
                    print(f"Failed to delete log {log_file.name}: {e}")
            print("Cleaned logs directory.")
        except Exception as e:
            print(f"Failed to clean logs directory: {e}")

    # 3. Cache Directories
    dirs_to_clean = [
        "audio_cache",
        "scene_cache",
        "stem_cache",
        "trigger_cache",
        "video_cache",
        "thumbnails",
        "temp",
        "cache"
    ]

    for dir_name in dirs_to_clean:
        dir_path = root_dir / dir_name
        if dir_path.exists():
            try:
                # Remove entire directory tree
                shutil.rmtree(dir_path)
                # Recreate empty directory
                dir_path.mkdir(exist_ok=True)
                print(f"Reset directory: {dir_name}")
            except Exception as e:
                print(f"Failed to reset {dir_name}: {e}")
        else:
            # Ensure directory exists even if it didn't before
            try:
                dir_path.mkdir(exist_ok=True)
                print(f"Created directory: {dir_name}")
            except Exception as e:
                print(f"Failed to create {dir_name}: {e}")

    print("Startup cleanup complete.")
