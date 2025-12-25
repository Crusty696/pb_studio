#!/usr/bin/env python
"""
ONNX Runtime Stem Separation - ECHTER TEST
============================================
Testet die Stem-Separation mit einer echten Audio-Datei.
Erfolgskriterium: Alle Stems müssen die gleiche Länge wie das Original haben.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import soundfile as sf

from pb_studio.audio.stem_separator import StemSeparator
from pb_studio.utils.logger import get_logger

logger = get_logger(__name__)

# Test-Konfiguration
AUDIO_FILE = Path(r"C:\Users\david\Videos\Music-Video_Clips\AV\Audio\Psy-Set\Progressive Psy Summer Dream mix  by Crusty Free download.wav")
TOLERANCE_SECONDS = 1.0  # Erlaubte Abweichung in Sekunden


def get_duration(file_path: Path) -> float:
    """Gibt die Dauer einer Audio-Datei in Sekunden zurück."""
    info = sf.info(str(file_path))
    return info.duration


def progress_callback(stem: str, progress: float):
    """Fortschritts-Callback für die Stem-Separation."""
    print(f"  [{stem}] {progress:.1%}", end="\r", flush=True)


def main():
    print("=" * 70)
    print("ONNX RUNTIME STEM SEPARATION - ECHTER TEST")
    print("=" * 70)
    
    # 1. Prüfe Original-Datei
    if not AUDIO_FILE.exists():
        print(f"❌ FEHLER: Audio-Datei nicht gefunden: {AUDIO_FILE}")
        return 1
    
    original_duration = get_duration(AUDIO_FILE)
    print(f"\n📁 Original-Datei: {AUDIO_FILE.name}")
    print(f"   Dauer: {original_duration:.2f} Sekunden ({original_duration/60:.2f} Minuten)")
    
    # 2. Initialisiere Separator
    print("\n🔧 Initialisiere StemSeparator...")
    try:
        separator = StemSeparator(model_preset="kuielab")
        print(f"   DirectML aktiv: {separator.use_directml}")
        print(f"   Segment Size: {separator.stem_segment_size}")
        print(f"   Batch Size: {separator.stem_batch_size}")
    except Exception as e:
        print(f"❌ FEHLER bei Initialisierung: {e}")
        return 1
    
    # 3. Führe Stem-Separation durch
    print("\n🎵 Starte Stem-Separation...")
    print("   (Dies kann bei 55 Minuten Audio sehr lange dauern!)")
    
    start_time = time.time()
    try:
        result = separator.separate(
            audio_path=AUDIO_FILE,
            progress_callback=progress_callback
        )
    except Exception as e:
        print(f"\n❌ FEHLER bei Stem-Separation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    elapsed = time.time() - start_time
    print(f"\n\n✅ Separation abgeschlossen in {elapsed:.1f} Sekunden ({elapsed/60:.1f} Minuten)")
    
    # 4. Validiere Stem-Längen
    print("\n" + "=" * 70)
    print("VALIDIERUNG - STEM-LÄNGEN")
    print("=" * 70)
    print(f"\nOriginal-Dauer: {original_duration:.2f}s ({original_duration/60:.2f} min)")
    print(f"Toleranz: ±{TOLERANCE_SECONDS}s")
    print("-" * 70)
    
    all_passed = True
    for stem_name, stem_path in result.items():
        stem_path = Path(stem_path)
        if not stem_path.exists():
            print(f"❌ {stem_name}: Datei nicht gefunden!")
            all_passed = False
            continue
        
        stem_duration = get_duration(stem_path)
        diff = abs(stem_duration - original_duration)
        
        if diff <= TOLERANCE_SECONDS:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            all_passed = False
        
        print(f"{status} | {stem_name:8} | {stem_duration:.2f}s | Diff: {diff:.2f}s | {stem_path.name}")
    
    # 5. Finale Bewertung
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 TEST BESTANDEN: Alle Stems haben die korrekte Länge!")
        return 0
    else:
        print("❌ TEST FEHLGESCHLAGEN: Nicht alle Stems haben die korrekte Länge!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
