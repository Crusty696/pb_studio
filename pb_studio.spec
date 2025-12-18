# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Spec File für PB_studio

Erstellt standalone Windows .exe mit allen Dependencies.

Build-Anleitung:
    pyinstaller pb_studio.spec

Ausgabe:
    dist/PB_studio/PB_studio.exe (Windows Executable)

Author: PB_studio Development Team
Task: D2 - PyInstaller Konfiguration
"""

import os
import sys
from pathlib import Path

# ============================================================================
# Build Configuration
# ============================================================================

block_cipher = None

# Application Info
APP_NAME = 'PB_studio'
APP_VERSION = '1.0.0'
APP_AUTHOR = 'PB_studio Development Team'

# Paths
src_path = Path('src')
pb_studio_path = src_path / 'pb_studio'

# ============================================================================
# Hidden Imports
# ============================================================================

# PyQt6 + Qt Runtime
hiddenimports_pyqt = [
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.sip',
]

# Audio Analysis
hiddenimports_audio = [
    'librosa',
    'librosa.core',
    'librosa.feature',
    'librosa.beat',
    'librosa.decompose',
    'soundfile',
    'audioread',
    'resampy',
    'numba',
    'numba.core',
    'numba.typed',
    'scipy',
    'scipy.signal',
    'scipy.fft',
]

# Video Processing
hiddenimports_video = [
    'cv2',
    'PIL',
    'PIL.Image',
    'PIL.ImageDraw',
    'PIL.ImageFont',
]

# Machine Learning / AI
hiddenimports_ml = [
    'torch',
    'torchvision',
    'transformers',
    'clip',
    'demucs',
    'demucs.apply',
    'demucs.separate',
    'faiss',
    'faiss.swigfaiss',
]

# Database
hiddenimports_db = [
    'sqlalchemy',
    'sqlalchemy.orm',
    'sqlalchemy.ext.declarative',
]

# GUI Extensions
hiddenimports_gui = [
    'dearpygui',
    'dearpygui.dearpygui',
]

# Utilities
hiddenimports_utils = [
    'pathlib',
    'json',
    'xml.etree.ElementTree',
]

# All Hidden Imports
hiddenimports = (
    hiddenimports_pyqt +
    hiddenimports_audio +
    hiddenimports_video +
    hiddenimports_ml +
    hiddenimports_db +
    hiddenimports_gui +
    hiddenimports_utils
)

# ============================================================================
# Data Files (Icons, Resources, Models)
# ============================================================================

datas = []

# Icons
if (pb_studio_path / 'resources' / 'icons').exists():
    datas.append((str(pb_studio_path / 'resources' / 'icons'), 'resources/icons'))

# Pretrained Models (if exists)
if (pb_studio_path / 'models').exists():
    datas.append((str(pb_studio_path / 'models'), 'models'))

# Example Presets (if exists)
if (pb_studio_path / 'presets').exists():
    datas.append((str(pb_studio_path / 'presets'), 'presets'))

# ============================================================================
# Binary Dependencies
# ============================================================================

# FFmpeg binaries (for audio/video processing)
# NOTE: User muss FFmpeg separat installieren oder als Dependency packen

binaries = []
if os.path.exists('bin/ffmpeg.exe'):
    binaries.append(('bin/ffmpeg.exe', 'bin'))
    print("Found ffmpeg.exe, bundling...")
if os.path.exists('bin/ffprobe.exe'):
    binaries.append(('bin/ffprobe.exe', 'bin'))
    print("Found ffprobe.exe, bundling...")

# ============================================================================
# Analysis (Entry Point)
# ============================================================================

a = Analysis(
    ['src/pb_studio/__main__.py'],  # Entry point
    pathex=[str(src_path)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',  # Exclude unused GUI frameworks
        'matplotlib',  # Exclude if not used
        'tensorflow',
        'tensorboard',
        'keras',
        'notebook',
        'ipython',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    module_collection_mode={
        'tensorboard': 'exclude',
        'tensorflow': 'exclude',
        'keras': 'exclude',
        'notebook': 'exclude',
        'ipython': 'exclude',
    },
)

# ============================================================================
# PYZ Archive (Python Files)
# ============================================================================

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

# ============================================================================
# Executable
# ============================================================================

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # UPX compression (smaller .exe)
    console=False,  # No console window (GUI app)
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='src/pb_studio/resources/icons/app_icon.ico',
)

# ============================================================================
# Collect (Bundle alle Files)
# ============================================================================

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)

# ============================================================================
# Build Info
# ============================================================================

print(f"""
================================================================================
PyInstaller Build Configuration: {APP_NAME}
================================================================================

Version:        {APP_VERSION}
Author:         {APP_AUTHOR}
Entry Point:    src/pb_studio/__main__.py
Output:         dist/{APP_NAME}/{APP_NAME}.exe

Hidden Imports: {len(hiddenimports)} modules
Data Files:     {len(datas)} directories
Binaries:       {len(binaries)} files

Build Command:  pyinstaller pb_studio.spec

================================================================================
""")
