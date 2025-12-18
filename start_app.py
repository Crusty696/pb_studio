"""
Einfacher App-Starter für PB_studio
"""
import os
import sys

# Workaround für ältere GPUs: NVFuser deaktivieren, um DLL-Load-Fehler (nvfuser_codegen.dll) zu vermeiden
os.environ.setdefault("PYTORCH_JIT_USE_NNC_NOT_NVFUSER", "1")
os.environ.setdefault("PYTORCH_NVFUSER_DISABLE", "1")
os.environ.setdefault("TORCH_NVFUSER_DISABLE", "1")

# Setze PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Starte die App
from PyQt6.QtWidgets import QApplication

from pb_studio.gui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
