import os
import shutil
import glob

def remove_directory(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            print(f"[DELETED] Directory: {path}")
        except Exception as e:
            print(f"[ERROR] Could not delete directory {path}: {e}")

def remove_pattern(pattern):
    for filepath in glob.glob(pattern, recursive=True):
        if os.path.isfile(filepath):
            try:
                os.remove(filepath)
                print(f"[DELETED] File: {filepath}")
            except Exception as e:
                print(f"[ERROR] Could not delete file {filepath}: {e}")

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Cleaning up in: {root_dir}")
    
    # 1. Standard Python/Test Caches
    print("\n--- Removing Standard Caches ---")
    remove_pattern(os.path.join(root_dir, "**", ".pytest_cache")) # Directories actually
    remove_pattern(os.path.join(root_dir, "**", "__pycache__"))
    
    # Better way for directories than glob with recursive=True which finds files usually? 
    # Let's walk for __pycache__ and .pytest_cache
    for root, dirs, files in os.walk(root_dir):
        for d in dirs[:]:
            if d == "__pycache__" or d == ".pytest_cache" or d == ".mypy_cache":
                path = os.path.join(root, d)
                remove_directory(path)
                dirs.remove(d)

    # 2. Application Specific Caches (Root Level)
    print("\n--- Removing Application Caches ---")
    cache_dirs = [
        "audio_cache",
        "video_cache",
        "scene_cache",
        "stem_cache",
        "trigger_cache",
        "thumbnails",
        "temp",
        "logs",
        "output" # Maybe? The user said "alles was nicht für die app gebraucht wird". Generated output is arguably waiting to be deleted.
    ]
    
    for relative_path in cache_dirs:
        full_path = os.path.join(root_dir, relative_path)
        if os.path.exists(full_path):
            # Clean content but keep directory? Or delete fully?
            # User said "alles ... weg". Often better to recreate locally.
            remove_directory(full_path)
            # Recreate empty to avoid 'not found' errors immediately on partial start
            os.makedirs(full_path, exist_ok=True)
            print(f"[RECREATED] Directory (Empty): {full_path}")

    # 3. Files in Root
    print("\n--- Removing Root Files ---")
    patterns = [
        "*.log",
        "project.db",
        "project.db-shm",
        "project.db-wal",
        "fault_dump.log"
    ]
    
    for pat in patterns:
        for f in glob.glob(os.path.join(root_dir, pat)):
            try:
                os.remove(f)
                print(f"[DELETED] File: {f}")
            except Exception as e:
                print(f"[ERROR] Could not delete {f}: {e}")

    print("\nCleanup Complete.")

if __name__ == "__main__":
    main()
