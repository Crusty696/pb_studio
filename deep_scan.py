"""
DEEP DEPENDENCY SCANNER
Scans for specific function calls that imply hidden/optional dependencies.
"""
import os
import re
from pathlib import Path

# Patterns implying extra dependencies
PATTERNS = {
    'opencv-contrib-python': [
        r'cv2\.xfeatures2d', r'cv2\.face', r'cv2\.tracking', r'cv2\.aruco', 
        r'cv2\.bgsegm', r'cv2\.bioinspired', r'cv2\.cuda', r'cv2\.img_hash', 
        r'cv2\.optflow', r'cv2\.plot', r'cv2\.saliency', r'cv2\.text', 
        r'cv2\.videostab', r'cv2\.ximgproc', r'cv2\.xphoto'
    ],
    'resampy': [r'librosa\.resample'],  # Often needs resampy for high quality
    'pydantic-settings': [r'BaseSettings', r'pydantic_settings'],
    'email-validator': [r'EmailStr'],
    'aiosqlite': [r'sqlite\+aiosqlite'],
    'asyncpg': [r'postgresql\+asyncpg'],
    'psycopg2': [r'postgresql\+psycopg2'],
    'scikit-learn': [r'sklearn\.'],  # Common companion to scipy/numpy
    'matplotlib': [r'matplotlib\.', r'plt\.'],  # Plotting often missed
    'pandas': [r'pandas', r'\.DataFrame', r'pd\.read_csv'],
    'openpyxl': [r'pd\.read_excel', r'\.to_excel'],
}

def scan_file(filepath):
    try:
        content = Path(filepath).read_text(encoding='utf-8')
    except:
        return []
    
    found = []
    for package, patterns in PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, content):
                found.append(f"{package} (Trigger: {pattern})")
    return found

def main():
    src_dir = Path('src')
    results = {}
    
    print("SCANNING FOR HIDDEN DEPENDENCIES...")
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                path = Path(root) / file
                hits = scan_file(path)
                if hits:
                    results[str(path)] = hits

    # Report
    all_reqs = set()
    print("\nRESULTS:")
    for path, hits in results.items():
        # print(f"{path}:") 
        for hit in hits:
            # print(f"  - {hit}")
            all_reqs.add(hit.split(' (')[0])
            
    print("\nPOTENTIAL MISSING PACKAGES:")
    for pkg in sorted(all_reqs):
        print(f"  - {pkg}")

if __name__ == '__main__':
    main()
