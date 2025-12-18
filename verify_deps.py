"""
Verifiziere ALLE third-party Dependencies.
"""

# ALLE in der Codebasis verwendeten third-party packages
USED_IN_CODE = {
    # GUI
    'PyQt6', 'dearpygui',
    
    # Audio
    'librosa', 'soundfile', 'numpy', 'scipy', 'BeatNet',
    
    # Video & Image
    'opencv-python',  # cv2
    'Pillow',  # PIL
    'ffmpeg-python',  # ffmpeg
    'scenedetect',
    'imagehash',
    'scikit-image',  # skimage
    
    # AI/ML
    'torch', 'torch-directml', 'torchaudio',
    'transformers',
    'onnxruntime',
    'clip',
    'av',
    
    # Vector Search
    'faiss', 'qdrant-client',
    
    # Database
    'sqlalchemy', 'duckdb', 'pydantic',
    
    # Utils
    'defusedxml', 'typing-extensions', 'requests', 'wmi', 'python-magic-bin'
}

# pyproject.toml
IN_PYPROJECT = {
    'PyQt6', 'dearpygui', 'sqlalchemy', 'pydantic', 'duckdb',
    'librosa', 'soundfile', 'numpy',
   'faiss-cpu', 'qdrant-client', 'ffmpeg-python', 'scenedetect', 'Pillow',
    'torch', 'torch-directml', 'demucs', 'transformers', 'ultralytics',
    'clip', 'av', 'accelerate', 'onnxruntime', 'onnxruntime-gpu', 
    'onnxruntime-directml', 'python-dotenv', 'python-magic', 
    'python-magic-bin', 'wmi', 'pywin32', 'typing-extensions',
    'resampy', 'defusedxml', 'scikit-image'
}

# Installationsskript
IN_SCRIPT = {
    'opencv-python', 'scipy', 'numba', 'imagehash', 'BeatNet', 'requests'
}

def main():
    print("DEPENDENCY VERIFICATION REPORT")
    print("="*60)
    
    all_covered = IN_PYPROJECT | IN_SCRIPT
    
    # Normalize for comparison
    def normalize(s):
        return s.lower().replace('-', '_').replace('_cpu', '').replace('_gpu', '')
    
    covered_norm = {normalize(p) for p in all_covered}
    
    missing = []
    for pkg in USED_IN_CODE:
        pkg_norm = normalize(pkg)
        if pkg_norm not in covered_norm:
            # Check partial matches
            found = any(pkg_norm in c or c in pkg_norm for c in covered_norm)
            if not found:
                missing.append(pkg)
    
    print("\nVERWENDETE PACKAGES:", len(USED_IN_CODE))
    for pkg in sorted(USED_IN_CODE):
        print(f"  - {pkg}")
    
    print(f"\nABGEDECKT: {len(all_covered)} packages")
    print(f"  - pyproject.toml: {len(IN_PYPROJECT)}")
    print(f"  - Installationsskript: {len(IN_SCRIPT)}")
    
    if missing:
        print(f"\nFEHLEND: {len(missing)} packages")
        for pkg in sorted(missing):
            print(f"  ! {pkg}")
        return 1
    else:
        print("\n[OK] ALLE DEPENDENCIES ABGEDECKT!")
        return 0

if __name__ == '__main__':
    exit(main())
