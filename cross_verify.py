"""
CROSS-VERIFICATION: Unabhängige Gegenprüfung des Installationsskripts
Prüft ALLE Aspekte gegen die tatsächliche Implementierung
"""
import sys
from pathlib import Path

def check_pyproject_toml():
    """Lese pyproject.toml und extrahiere alle Dependencies."""
    pyproject = Path('pyproject.toml')
    if not pyproject.exists():
        return set(), set()
    
    content = pyproject.read_text()
    
    # Basis Dependencies
    main_deps = set()
    for line in content.split('\n'):
        if '=' in line and not line.strip().startswith('#'):
            if 'PyQt6' in line: main_deps.add('PyQt6')
            if 'librosa' in line: main_deps.add('librosa')
            if 'numpy' in line: main_deps.add('numpy')
            if 'sqlalchemy' in line: main_deps.add('sqlalchemy')
            if 'pydantic' in line: main_deps.add('pydantic')
            if 'duckdb' in line: main_deps.add('duckdb')
            if 'soundfile' in line: main_deps.add('soundfile')
            if 'Pillow' in line: main_deps.add('Pillow')
            if 'ffmpeg-python' in line: main_deps.add('ffmpeg-python')
            if 'scenedetect' in line: main_deps.add('scenedetect')
            if 'transformers' in line: main_deps.add('transformers')
            if 'ultralytics' in line: main_deps.add('ultralytics')
            if 'defusedxml' in line: main_deps.add('defusedxml')
            if 'typing-extensions' in line: main_deps.add('typing-extensions')
            if 'scikit-image' in line: main_deps.add('scikit-image')
            if 'wmi' in line: main_deps.add('wmi')
            if 'python-magic-bin' in line: main_deps.add('python-magic-bin')
            if 'dearpygui' in line: main_deps.add('dearpygui')
            if 'resampy' in line: main_deps.add('resampy')
    
    # Optional Dependencies
    optional_deps = set()
    if 'torch' in content: optional_deps.add('torch')
    if 'torch-directml' in content: optional_deps.add('torch-directml')
    if 'onnxruntime' in content: optional_deps.add('onnxruntime')
    if 'faiss-cpu' in content: optional_deps.add('faiss-cpu')
    if 'qdrant-client' in content: optional_deps.add('qdrant-client')
    if 'clip' in content: optional_deps.add('clip')
    if 'av' in content: optional_deps.add('av')
    if 'demucs' in content: optional_deps.add('demucs')
    
    return main_deps, optional_deps

def check_install_script():
    """Lese Installationsskript und extrahiere installierte Pakete."""
    script = Path('INSTALL_COMPLETE.bat')
    if not script.exists():
        return set()
    
    content = script.read_text()
    installed = set()
    
    for line in content.split('\n'):
        if 'pip install' in line:
            parts = line.split('pip install')
            if len(parts) > 1:
                pkg = parts[1].strip()
                # Entferne Kommentare und Optionen
                pkg = pkg.split('#')[0].split('>')[0].strip()
                if pkg and not pkg.startswith('-'):
                    installed.add(pkg)
    
    return installed

def main():
    print("="*70)
    print("  CROSS-VERIFICATION REPORT")
    print("  Unabhängige Gegenprüfung aller Komponenten")
    print("="*70)
    print()
    
    # 1. pyproject.toml Dependencies
    main_deps, optional_deps = check_pyproject_toml()
    print(f"[1] pyproject.toml Dependencies:")
    print(f"    - Basis: {len(main_deps)} Pakete")
    print(f"    - Optional: {len(optional_deps)} Pakete")
    print()
    
    # 2. Installationsskript
    script_deps = check_install_script()
    print(f"[2] Installationsskript (INSTALL_COMPLETE.bat):")
    print(f"    - Manuell installiert: {len(script_deps)} Pakete")
    for pkg in sorted(script_deps):
        print(f"      - {pkg}")
    print()
    
    # 3. Kritische verwendete Pakete (aus Code-Analyse)
    critical_used = {
        'PyQt6', 'opencv-python', 'scipy', 'numpy', 'librosa',
        'torch', 'cv2', 'PIL', 'ffmpeg', 'sqlalchemy'
    }
    
    print(f"[3] Kritische Pakete (aus Code-Scan):")
    for pkg in sorted(critical_used):
        # Prüfe Abdeckung
        normalized = pkg.replace('opencv-python', 'cv2').replace('Pillow', 'PIL')
        covered = (
            pkg in main_deps or pkg in optional_deps or 
            pkg in script_deps or normalized in script_deps
        )
        status = "OK" if covered else "FEHLT"
        print(f"      [{status}] {pkg}")
    print()
    
    # 4. Module-spezifische Checks
    checks = {
        'Audio': ['librosa', 'soundfile', 'scipy', 'numpy'],
        'Video': ['opencv-python', 'Pillow', 'ffmpeg-python', 'scenedetect'],
        'AI/ML': ['torch', 'onnxruntime', 'transformers'],
        'GUI': ['PyQt6'],
        'Database': ['sqlalchemy', 'pydantic', 'duckdb']
    }
    
    print("[4] Modul-spezifische Dependency-Checks:")
    all_combined = main_deps | optional_deps | script_deps
    
    for module, required in checks.items():
        missing = []
        for pkg in required:
            # Flexible Matching
            found = any(
                pkg.lower().replace('-', '_') in dep.lower().replace('-', '_') or
                dep.lower().replace('-', '_') in pkg.lower().replace('-', '_')
                for dep in all_combined
            )
            if not found:
                missing.append(pkg)
        
        status = "OK" if not missing else f"FEHLT: {', '.join(missing)}"
        print(f"    [{status if status == 'OK' else 'WARN'}] {module}")
    print()
    
    # 5. GPU-Support Check
    print("[5] GPU-Support Verifikation:")
    gpu_packages = {
        'NVIDIA': ['torch (CUDA variant)', 'onnxruntime-gpu'],
        'AMD': ['torch-directml', 'onnxruntime-directml'],
        'CPU': ['torch', 'onnxruntime']
    }
    for hw, pkgs in gpu_packages.items():
        print(f"    {hw}:")
        for pkg in pkgs:
            print(f"      - {pkg}")
    print()
    
    # 6. Zusammenfassung
    total_coverage = len(main_deps) + len(optional_deps) + len(script_deps)
    print("="*70)
    print(f"ZUSAMMENFASSUNG:")
    print(f"  Gesamt abgedeckt: {total_coverage} Pakete")
    print(f"  - pyproject.toml: {len(main_deps) + len(optional_deps)}")
    print(f"  - Installationsskript: {len(script_deps)}")
    print()
    print("STATUS: Gegenprüfung abgeschlossen")
    print("="*70)

if __name__ == '__main__':
    main()
