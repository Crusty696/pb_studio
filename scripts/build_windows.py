"""
Automated Build Script für PB_studio Windows .exe

Führt vollautomatisch aus:
1. Virtuelle Umgebung aktivieren
2. Dependencies installieren
3. PyInstaller Build ausführen
4. Build-Verzeichnis aufräumen
5. Erfolgs-/Fehler-Reporting

Usage:
    python scripts/build_windows.py

Output:
    dist/PB_studio/PB_studio.exe

Author: PB_studio Development Team
Task: D3 - Build-Script
"""

import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ============================================================================
# Build Configuration
# ============================================================================

APP_NAME = "PB_studio"
SPEC_FILE = "pb_studio.spec"
VENV_DIR = ".venv"
BUILD_DIR = "build"
DIST_DIR = "dist"

# Colors für Terminal-Output (Windows & Unix)
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"


# ============================================================================
# Helper Functions
# ============================================================================


def print_header(message: str):
    """Print formatted header."""
    print(f"\n{COLOR_BLUE}{'=' * 80}{COLOR_RESET}")
    print(f"{COLOR_BLUE}{message.center(80)}{COLOR_RESET}")
    print(f"{COLOR_BLUE}{'=' * 80}{COLOR_RESET}\n")


def print_success(message: str):
    """Print success message."""
    print(f"{COLOR_GREEN}✓ {message}{COLOR_RESET}")


def print_error(message: str):
    """Print error message."""
    print(f"{COLOR_RED}✗ {message}{COLOR_RESET}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{COLOR_YELLOW}⚠ {message}{COLOR_RESET}")


def print_info(message: str):
    """Print info message."""
    print(f"{COLOR_BLUE}ℹ {message}{COLOR_RESET}")


def run_command(cmd: list, description: str, cwd: Path = None) -> bool:
    """
    Run shell command and return success status.

    Args:
        cmd: Command as list (e.g., ["python", "-m", "pip", "install", "..."])
        description: Human-readable description
        cwd: Working directory (optional)

    Returns:
        True if command succeeded, False otherwise
    """
    print_info(f"{description}...")

    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, shell=True if os.name == "nt" else False
        )

        if result.returncode == 0:
            print_success(f"{description} - Erfolg")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print_error(f"{description} - Fehler")
            if result.stderr:
                print(result.stderr)
            return False

    except Exception as e:
        print_error(f"{description} - Exception: {e}")
        return False


# ============================================================================
# Build Steps
# ============================================================================


def check_prerequisites() -> bool:
    """Check if all prerequisites are met."""
    print_header("Voraussetzungen prüfen")

    # Check spec file
    if not Path(SPEC_FILE).exists():
        print_error(f"Spec-Datei nicht gefunden: {SPEC_FILE}")
        return False
    print_success(f"Spec-Datei gefunden: {SPEC_FILE}")

    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print_success(f"Python Version: {py_version}")

    # Check if running in virtual environment (recommended)
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )

    if in_venv:
        print_success("Virtuelle Umgebung aktiv")
    else:
        print_warning("WARNUNG: Keine virtuelle Umgebung aktiv!")
        print_warning("Empfohlen: Aktiviere .venv vor dem Build")

    return True


def install_dependencies() -> bool:
    """Install required dependencies."""
    print_header("Dependencies installieren")

    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print_warning("requirements.txt nicht gefunden - überspringe")
        return True

    # Install dependencies
    cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    return run_command(cmd, "pip install requirements.txt")


def install_pyinstaller() -> bool:
    """Ensure PyInstaller is installed."""
    print_header("PyInstaller prüfen/installieren")

    # Check if PyInstaller is installed
    try:
        import PyInstaller

        print_success(f"PyInstaller bereits installiert: {PyInstaller.__version__}")
        return True
    except ImportError:
        print_info("PyInstaller nicht gefunden - installiere...")
        cmd = [sys.executable, "-m", "pip", "install", "pyinstaller"]
        return run_command(cmd, "pip install pyinstaller")


def clean_build_dirs():
    """Clean build and dist directories."""
    print_header("Build-Verzeichnisse aufräumen")

    for dirname in [BUILD_DIR, DIST_DIR]:
        dirpath = Path(dirname)
        if dirpath.exists():
            print_info(f"Lösche: {dirname}/")
            shutil.rmtree(dirpath)
            print_success(f"{dirname}/ gelöscht")
        else:
            print_info(f"{dirname}/ existiert nicht - überspringe")


def run_pyinstaller() -> bool:
    """Run PyInstaller build."""
    print_header("PyInstaller Build starten")

    # Build command
    cmd = ["pyinstaller", SPEC_FILE, "--clean", "--noconfirm"]

    print_info(f"Befehl: {' '.join(cmd)}")
    print_info("⏳ Build läuft... (kann mehrere Minuten dauern)")

    start_time = time.time()
    success = run_command(cmd, "PyInstaller Build")
    elapsed = time.time() - start_time

    if success:
        print_success(f"Build abgeschlossen in {elapsed:.1f} Sekunden")

    return success


def verify_build() -> bool:
    """Verify that build was successful."""
    print_header("Build-Ergebnis verifizieren")

    # Check if executable exists
    exe_path = Path(DIST_DIR) / APP_NAME / f"{APP_NAME}.exe"

    if not exe_path.exists():
        print_error(f"Executable nicht gefunden: {exe_path}")
        return False

    print_success(f"Executable gefunden: {exe_path}")

    # Check file size
    size_mb = exe_path.stat().st_size / (1024 * 1024)
    print_info(f"Größe: {size_mb:.2f} MB")

    # List all files in dist directory
    print_info(f"\nInhalt von {DIST_DIR}/{APP_NAME}/:")
    for item in exe_path.parent.iterdir():
        if item.is_file():
            item_size = item.stat().st_size / (1024 * 1024)
            print(f"  - {item.name} ({item_size:.2f} MB)")
        elif item.is_dir():
            print(f"  - {item.name}/ (Verzeichnis)")

    return True


def create_build_info():
    """Create build info file."""
    print_header("Build-Info erstellen")

    build_info = {
        "app_name": APP_NAME,
        "build_date": datetime.now().isoformat(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": sys.platform,
    }

    # Write to dist directory
    info_path = Path(DIST_DIR) / APP_NAME / "BUILD_INFO.txt"

    try:
        with open(info_path, "w", encoding="utf-8") as f:
            f.write("PB_studio Build Information\n")
            f.write(f"{'=' * 40}\n\n")
            for key, value in build_info.items():
                f.write(f"{key}: {value}\n")

        print_success(f"Build-Info geschrieben: {info_path}")
        return True
    except Exception as e:
        print_warning(f"Build-Info konnte nicht erstellt werden: {e}")
        return False


# ============================================================================
# Main Build Process
# ============================================================================


def main():
    """Main build process."""
    print_header(f"PB_studio Windows Build - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    # Step 1: Check prerequisites
    if not check_prerequisites():
        print_error("\n❌ Voraussetzungen nicht erfüllt - Abbruch")
        return 1

    # Step 2: Install dependencies
    if not install_dependencies():
        print_error("\n❌ Dependency-Installation fehlgeschlagen - Abbruch")
        return 1

    # Step 3: Install PyInstaller
    if not install_pyinstaller():
        print_error("\n❌ PyInstaller-Installation fehlgeschlagen - Abbruch")
        return 1

    # Step 4: Clean build directories
    clean_build_dirs()

    # Step 5: Run PyInstaller
    if not run_pyinstaller():
        print_error("\n❌ PyInstaller Build fehlgeschlagen - Abbruch")
        return 1

    # Step 6: Verify build
    if not verify_build():
        print_error("\n❌ Build-Verifikation fehlgeschlagen - Abbruch")
        return 1

    # Step 7: Create build info
    create_build_info()

    # Success!
    elapsed = time.time() - start_time

    print_header("✅ BUILD ERFOLGREICH!")
    print_success(f"Gesamtdauer: {elapsed:.1f} Sekunden")
    print_success(f"Executable: dist/{APP_NAME}/{APP_NAME}.exe")
    print_info("\nNächste Schritte:")
    print_info(f"  1. Teste die .exe: dist/{APP_NAME}/{APP_NAME}.exe")
    print_info("  2. Erstelle Installer mit Inno Setup (Task D4)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
