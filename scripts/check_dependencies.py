#!/usr/bin/env python3
"""
PB Studio - Umfassendes Abhängigkeits-Prüfskript
=================================================
Überprüft alle externen Bibliotheken und ML-Modelle vor dem Start der Anwendung.

Autor: PB Studio Team
Datum: 25.12.2025
"""

import sys
import os
import subprocess
import platform
from pathlib import Path
from typing import Optional, Tuple
import importlib.util
import logging

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# FARBCODES FÜR KONSOLENAUSGABE (Windows-kompatibel)
# ==============================================================================
class Colors:
    """ANSI-Farbcodes für Konsolenausgabe."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    GRAY = '\033[90m'

    @staticmethod
    def enable_windows_colors():
        """Aktiviert ANSI-Farben auf Windows."""
        if platform.system() == 'Windows':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except Exception:
                pass


# Farben aktivieren
Colors.enable_windows_colors()


# ==============================================================================
# HELPER FUNKTIONEN
# ==============================================================================

def check_module(module_name: str, package_name: Optional[str] = None) -> Tuple[bool, str]:
    """
    Prüft ob ein Python-Modul importierbar ist.
    
    Args:
        module_name: Name des zu importierenden Moduls
        package_name: Optionaler Paketname (falls abweichend)
    
    Returns:
        Tuple (success: bool, version_or_error: str)
    """
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return False, "Nicht gefunden"
        
        # Versuche Version zu ermitteln
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 
                            getattr(module, 'VERSION', 'Version unbekannt'))
            return True, str(version)
        except Exception as e:
            return True, f"Import-Warnung: {e}"
            
    except ModuleNotFoundError:
        return False, "Nicht installiert"
    except Exception as e:
        return False, str(e)


def check_command(command: str) -> Tuple[bool, str]:
    """
    Prüft ob ein Kommandozeilen-Tool verfügbar ist.
    
    Args:
        command: Name des Befehls
    
    Returns:
        Tuple (success: bool, version_or_error: str)
    """
    try:
        result = subprocess.run(
            [command, '-version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        # FFmpeg gibt Version in stderr aus
        output = result.stdout or result.stderr
        # Erste Zeile der Ausgabe
        version = output.strip().split('\n')[0] if output else 'Version unbekannt'
        return True, version[:60]  # Kürzen
    except FileNotFoundError:
        return False, "Nicht im PATH gefunden"
    except Exception as e:
        return False, str(e)


def print_header(text: str):
    """Druckt einen formatierten Header."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}  {text}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'=' * 60}{Colors.RESET}")


def print_section(text: str):
    """Druckt eine Sektion."""
    print(f"\n{Colors.BOLD}▸ {text}{Colors.RESET}")
    print(f"  {'-' * 40}")


def print_result(name: str, success: bool, detail: str, optional: bool = False):
    """
    Druckt ein Prüfergebnis.
    
    Args:
        name: Name der Abhängigkeit
        success: Erfolgsstatus
        detail: Version oder Fehlermeldung
        optional: Ob die Abhängigkeit optional ist
    """
    # Flush um Race Conditions zu vermeiden
    sys.stdout.flush()
    
    if success:
        icon = f"{Colors.GREEN}✓{Colors.RESET}"
        status = f"{Colors.GREEN}OK{Colors.RESET}"
    elif optional:
        icon = f"{Colors.YELLOW}○{Colors.RESET}"
        status = f"{Colors.YELLOW}Optional{Colors.RESET}"
    else:
        icon = f"{Colors.RED}✗{Colors.RESET}"
        status = f"{Colors.RED}FEHLT{Colors.RESET}"
    
    # Formatierung für Alignment
    print(f"  {icon} {name:<25} [{status}] {Colors.GRAY}{detail[:35]}{Colors.RESET}")


# ==============================================================================
# PRÜFUNGSMODULE
# ==============================================================================

def check_python_environment() -> bool:
    """Prüft Python-Umgebung."""
    print_section("Python-Umgebung")
    
    all_ok = True
    
    # Python-Version
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    version_ok = version.major == 3 and 10 <= version.minor < 13
    print_result("Python-Version", version_ok, version_str)
    if not version_ok:
        all_ok = False
    
    # venv aktiv?
    in_venv = sys.prefix != sys.base_prefix
    print_result("Virtuelle Umgebung", in_venv, 
                 os.path.basename(sys.prefix) if in_venv else "Nicht aktiv")
    
    # Plattform
    print_result("Plattform", True, f"{platform.system()} {platform.machine()}")
    
    return all_ok


def check_core_libraries() -> Tuple[int, int]:
    """
    Prüft Core-Bibliotheken.
    
    Returns:
        Tuple (erfolgreiche, fehlende)
    """
    print_section("Core-Bibliotheken (Kritisch)")
    
    libraries = [
        ("PyQt6", "PyQt6", False),
        ("SQLAlchemy", "sqlalchemy", False),
        ("Pydantic", "pydantic", False),
        ("ffmpeg-python", "ffmpeg", False),
        ("opencv-python", "cv2", False),
        ("Pillow", "PIL", False),
        ("numpy", "numpy", False),
    ]
    
    success = 0
    failed = 0
    
    for display_name, module_name, optional in libraries:
        ok, detail = check_module(module_name)
        print_result(display_name, ok, detail, optional)
        if ok:
            success += 1
        else:
            failed += 1
    
    return success, failed


def check_audio_libraries() -> Tuple[int, int]:
    """
    Prüft Audio-Analyse-Bibliotheken.
    
    Returns:
        Tuple (erfolgreiche, fehlende/optionale)
    """
    print_section("Audio-Analyse")
    
    libraries = [
        ("librosa", "librosa", False),
        ("soundfile", "soundfile", False),
        ("scipy", "scipy", False),
        ("audio-separator", "audio_separator", True),  # Optional wegen Stem-Separation
        ("resampy", "resampy", True),
    ]
    
    success = 0
    issues = 0
    
    for display_name, module_name, optional in libraries:
        ok, detail = check_module(module_name)
        print_result(display_name, ok, detail, optional)
        if ok:
            success += 1
        else:
            issues += 1
    
    return success, issues


def check_ai_ml_stack() -> Tuple[int, int, str]:
    """
    Prüft AI/ML-Stack mit Hardware-Erkennung.
    
    Returns:
        Tuple (erfolgreiche, fehlende, erkannter_hardware_typ)
    """
    print_section("AI/ML-Stack")
    
    success = 0
    issues = 0
    hardware_type = "CPU"
    
    # Basis-Bibliotheken
    base_libs = [
        ("transformers", "transformers", False),
        ("ultralytics", "ultralytics", False),
    ]
    
    for display_name, module_name, optional in base_libs:
        ok, detail = check_module(module_name)
        print_result(display_name, ok, detail, optional)
        if ok:
            success += 1
        else:
            issues += 1
    
    # ONNX Runtime Varianten prüfen
    print(f"\n  {Colors.GRAY}--- ONNX Runtime ---{Colors.RESET}")
    
    # CUDA prüfen
    cuda_ok, cuda_detail = check_module("onnxruntime")
    
    # Prüfe auf GPU-Provider
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        
        if 'CUDAExecutionProvider' in providers:
            hardware_type = "NVIDIA (CUDA)"
            print_result("onnxruntime-gpu", True, f"CUDA verfügbar", False)
            success += 1
        elif 'DmlExecutionProvider' in providers:
            hardware_type = "AMD/Intel (DirectML)"
            print_result("onnxruntime-directml", True, f"DirectML verfügbar", False)
            success += 1
        elif cuda_ok:
            print_result("onnxruntime (CPU)", True, cuda_detail, False)
            success += 1
    except ImportError:
        print_result("onnxruntime", False, "Nicht installiert", False)
        issues += 1
    except Exception as e:
        print_result("onnxruntime", False, str(e), False)
        issues += 1
    
    # PyTorch prüfen
    print(f"\n  {Colors.GRAY}--- PyTorch ---{Colors.RESET}")
    
    torch_ok, torch_detail = check_module("torch")
    if torch_ok:
        try:
            import torch
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                print_result("PyTorch + CUDA", True, f"CUDA {cuda_version}", False)
                hardware_type = "NVIDIA (CUDA)"
            else:
                # DirectML prüfen
                directml_ok, _ = check_module("torch_directml")
                if directml_ok:
                    print_result("PyTorch + DirectML", True, torch_detail, False)
                    hardware_type = "AMD/Intel (DirectML)"
                else:
                    print_result("PyTorch (CPU)", True, torch_detail, False)
            success += 1
        except Exception as e:
            print_result("PyTorch", True, f"Import-Warnung: {e}", False)
            success += 1
    else:
        print_result("PyTorch", False, torch_detail, True)
        issues += 1
    
    return success, issues, hardware_type


def check_ml_models() -> Tuple[int, int]:
    """
    Prüft ML-Modelle unter data/ai_models/.
    
    Returns:
        Tuple (gefundene, fehlende)
    """
    print_section("ML-Modelle")
    
    # Pfad zum Projektroot ermitteln
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    models_dir = project_root / "data" / "ai_models"
    
    models = [
        ("Moondream2 (ONNX)", "moondream2", True),
        ("Phi-3 Mini (DirectML)", "phi-3-mini-4k-directml", True),
        ("YOLOv8n (ONNX)", "yolov8n.onnx", True),
    ]
    
    found = 0
    missing = 0
    
    if not models_dir.exists():
        print(f"  {Colors.YELLOW}! Modell-Verzeichnis nicht gefunden: {models_dir}{Colors.RESET}")
        return 0, len(models)
    
    for display_name, model_path, optional in models:
        full_path = models_dir / model_path
        exists = full_path.exists()
        
        if exists:
            if full_path.is_dir():
                # Zähle ONNX-Dateien im Verzeichnis rekursiv
                onnx_files = list(full_path.rglob("*.onnx"))
                detail = f"Ordner: {len(onnx_files)} ONNX-Dateien"
            else:
                size_mb = full_path.stat().st_size / (1024 * 1024)
                detail = f"{size_mb:.1f} MB"
            found += 1
        else:
            detail = "Nicht vorhanden"
            missing += 1
        
        print_result(display_name, exists, detail, optional)
    
    return found, missing


def check_external_tools() -> Tuple[int, int]:
    """
    Prüft externe Tools (ffmpeg etc.).
    
    Returns:
        Tuple (gefundene, fehlende)
    """
    print_section("Externe Tools")
    
    tools = [
        ("FFmpeg", "ffmpeg", False),
    ]
    
    found = 0
    missing = 0
    
    for display_name, command, optional in tools:
        ok, detail = check_command(command)
        print_result(display_name, ok, detail, optional)
        if ok:
            found += 1
        else:
            missing += 1
    
    return found, missing


def check_vector_search() -> Tuple[int, int]:
    """
    Prüft Vector-Search-Bibliotheken.
    
    Returns:
        Tuple (gefundene, fehlende)
    """
    print_section("Vector Search")
    
    libraries = [
        ("faiss-cpu", "faiss", True),
        ("qdrant-client", "qdrant_client", True),
    ]
    
    found = 0
    missing = 0
    
    for display_name, module_name, optional in libraries:
        ok, detail = check_module(module_name)
        print_result(display_name, ok, detail, optional)
        if ok:
            found += 1
        else:
            missing += 1
    
    if found == 0:
        print(f"  {Colors.YELLOW}! Mindestens eine Vector-Search-Bibliothek empfohlen.{Colors.RESET}")
    
    return found, missing


# ==============================================================================
# HAUPTPROGRAMM
# ==============================================================================

def main() -> int:
    """
    Hauptfunktion des Abhängigkeits-Prüfskripts.
    
    Returns:
        Exit-Code (0 = OK, 1 = Fehler)
    """
    print_header("PB Studio - Abhängigkeits-Prüfung")
    print(f"  Zeitpunkt: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Alle Prüfungen durchführen
    python_ok = check_python_environment()
    
    core_ok, core_fail = check_core_libraries()
    audio_ok, audio_issues = check_audio_libraries()
    ai_ok, ai_issues, hardware = check_ai_ml_stack()
    models_ok, models_missing = check_ml_models()
    tools_ok, tools_missing = check_external_tools()
    vector_ok, vector_missing = check_vector_search()
    
    # Zusammenfassung
    print_header("ZUSAMMENFASSUNG")
    
    total_ok = core_ok + audio_ok + ai_ok + tools_ok
    total_fail = core_fail + tools_missing
    total_optional = audio_issues + ai_issues + models_missing + vector_missing
    
    print(f"\n  {Colors.BOLD}Erkannte Hardware:{Colors.RESET} {hardware}")
    print(f"\n  {Colors.GREEN}✓ Erfolgreich:{Colors.RESET}       {total_ok}")
    print(f"  {Colors.RED}✗ Kritisch fehlend:{Colors.RESET}  {total_fail}")
    print(f"  {Colors.YELLOW}○ Optional fehlend:{Colors.RESET}  {total_optional}")
    
    if models_ok == 0:
        print(f"\n  {Colors.YELLOW}! ML-Modelle fehlen - KI-Funktionen eingeschränkt.{Colors.RESET}")
        print(f"    Modelle herunterladen mit: python download_models.py")
    
    if total_fail > 0:
        print(f"\n  {Colors.RED}{Colors.BOLD}STATUS: FEHLER - Kritische Abhängigkeiten fehlen!{Colors.RESET}")
        print(f"  Bitte fehlende Bibliotheken installieren:")
        print(f"    pip install <bibliothek>")
        return 1
    else:
        print(f"\n  {Colors.GREEN}{Colors.BOLD}STATUS: OK - Alle kritischen Abhängigkeiten vorhanden.{Colors.RESET}")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Abgebrochen.{Colors.RESET}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Fehler: {e}{Colors.RESET}")
        sys.exit(1)
