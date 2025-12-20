"""
PB Studio - Automatischer Render-Test

Dieses Skript testet den vollständigen Workflow:
1. Audio-Analyse (BPM, Beatgrid)
2. Video-Clips importieren
3. Cut-List generieren
4. Video rendern (exakt so lang wie Audio)

Usage:
    python test_full_render.py --audio "pfad/zur/audio.wav" --videos "pfad/zum/video/ordner"
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pb_studio.audio.audio_analyzer import AudioAnalyzer
from pb_studio.pacing.advanced_pacing_engine import AdvancedPacingEngine, PacingMode
from pb_studio.pacing.pacing_models import CutListEntry
from pb_studio.video.video_renderer import VideoRenderer, RenderSettings
from pb_studio.utils.logger import setup_logging
from pb_studio.database.connection import get_db_manager
from pb_studio.database.models import Base

# Setup logging
logger = setup_logging(console_level=logging.INFO)


def get_audio_duration(audio_path: str) -> float:
    """Ermittle Audio-Dauer mit ffprobe."""
    import subprocess
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def collect_video_clips(video_dir: str, max_clips: int = 50) -> list[dict]:
    """Sammle Video-Clips aus einem Ordner."""
    clips = []
    video_extensions = {".mp4", ".mov", ".avi", ".mkv"}
    
    video_path = Path(video_dir)
    if not video_path.exists():
        logger.error(f"Video directory not found: {video_path}")
        return []

    for idx, file in enumerate(sorted(video_path.glob("*"))):
        if file.suffix.lower() in video_extensions:
            clips.append({
                "id": idx + 1,
                "file_path": str(file),
                "duration": 10.0,  # Wird später aktualisiert
                "analysis": {
                    "energy_score": 0.5,
                    "motion": {"motion_score": 0.5}
                }
            })
            if len(clips) >= max_clips:
                break
    
    logger.info(f"Gefunden: {len(clips)} Video-Clips")
    return clips


def run_full_render_test(
    audio_path: str,
    video_dir: str,
    output_path: str,
    duration_limit: float | None = None,
    use_gpu: bool = True
):
    """
    Führe vollständigen Render-Test durch.
    
    Args:
        audio_path: Pfad zur Audiodatei
        video_dir: Ordner mit Video-Clips
        output_path: Ausgabepfad für gerendertes Video
        duration_limit: Maximale Dauer in Sekunden (None = volle Länge)
        use_gpu: GPU-Beschleunigung verwenden
    """
    print("\n" + "="*60)
    print("  PB Studio - Vollständiger Render-Test")
    print("="*60 + "\n")

    # Initialize DB (needed for models)
    try:
        get_db_manager().init_db(Base)
    except Exception as e:
        logger.warning(f"Could not init DB: {e}")
    
    # 1. Audio-Dauer ermitteln
    print("[1/5] Ermittle Audio-Dauer...")
    audio_duration = get_audio_duration(audio_path)
    if audio_duration == 0.0:
        print("      FEHLER: Konnte Audio-Dauer nicht ermitteln.")
        return False

    target_duration = duration_limit if duration_limit else audio_duration
    print(f"      Audio-Dauer: {audio_duration:.2f}s ({audio_duration/60:.1f} min)")
    print(f"      Ziel-Dauer:  {target_duration:.2f}s ({target_duration/60:.1f} min)")
    
    # 2. Audio analysieren
    print("\n[2/5] Analysiere Audio (BPM, Beatgrid)...")
    analyzer = AudioAnalyzer()
    bpm_result = analyzer.analyze_bpm(audio_path)
    bpm = bpm_result["bpm"] if bpm_result else 140.0
    print(f"      BPM: {bpm:.1f}")
    
    # 3. Video-Clips sammeln
    print("\n[3/5] Sammle Video-Clips...")
    clips = collect_video_clips(video_dir)
    if not clips:
        print("      FEHLER: Keine Video-Clips gefunden!")
        return False
    
    # 4. Cut-List generieren
    print("\n[4/5] Generiere Cut-List...")
    engine = AdvancedPacingEngine()
    engine.pacing_mode = PacingMode.BEAT_SYNC
    
    def progress_callback(current, total, message):
        percent = (current / total * 100) if total > 0 else 0
        print(f"\r      Progress: {percent:.0f}% - {message}", end="", flush=True)
    
    cut_list_with_clips = engine.generate_cut_list_with_clips(
        audio_path=audio_path,
        available_clips=clips,
        expected_bpm=bpm,
        min_cut_interval=0.5,
        start_time=0,
        end_time=target_duration,
        progress_callback=progress_callback
    )
    print()  # Newline nach Progress
    
    # Convert to CutListEntry format
    cut_list = []
    for pacing_cut, clip_dict in cut_list_with_clips:
        start_time = float(pacing_cut.time)
        duration = 1.0 # default duration if not specified, engine usually calculates gaps

        # Calculate actual duration based on next cut or end time
        # This is a simplification for the test script

        entry = CutListEntry(
            clip_id=str(clip_dict.get("id", "unknown")), # Must be string
            start_time=start_time,
            end_time=start_time + duration, # Provisional end time
            metadata={
                "video_path": clip_dict.get("file_path", ""),
                "clip_in_point": 0.0,
                "clip_out_point": clip_dict.get("duration", 10.0),
                "trigger_type": pacing_cut.trigger_type,
                "energy": pacing_cut.strength
            }
        )
        cut_list.append(entry)
    
    # Fix end times to be contiguous
    for i in range(len(cut_list) - 1):
        cut_list[i].end_time = cut_list[i+1].start_time

    # Fix last cut end time
    if cut_list:
        cut_list[-1].end_time = target_duration

    print(f"      Generiert: {len(cut_list)} Schnitte")
    
    # 5. Video rendern
    print(f"\n[5/5] Rendere Video ({target_duration:.1f}s)...")
    settings = RenderSettings(
        use_gpu=use_gpu,
        gpu_encoder="auto",
        crf=23,
        preset="fast"
    )
    renderer = VideoRenderer(settings=settings)
    
    def render_progress(progress: float):
        print(f"\r      Render: {progress*100:.0f}%", end="", flush=True)
    
    try:
        result = renderer.render_video(
            cut_list=cut_list,
            audio_path=audio_path,
            output_path=output_path,
            progress_callback=render_progress
        )
        print()  # Newline nach Progress
        
        if result and Path(result).exists():
            output_size = Path(result).stat().st_size / (1024 * 1024)
            print(f"\n✅ ERFOLG!")
            print(f"   Output: {result}")
            print(f"   Größe:  {output_size:.1f} MB")

            # Verifiziere Output-Dauer
            output_duration = get_audio_duration(result)
            print(f"   Dauer:  {output_duration:.2f}s (Ziel: {target_duration:.2f}s)")

            duration_diff = abs(output_duration - target_duration)
            if duration_diff < 1.0:
                print(f"   ✅ Dauer-Check: OK (Differenz: {duration_diff:.2f}s)")
            else:
                print(f"   ⚠️ Dauer-Check: Differenz von {duration_diff:.2f}s!")

            return True
        else:
            print("\n❌ FEHLER: Rendering fehlgeschlagen!")
            return False
    except Exception as e:
        print(f"\n❌ CRASH: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PB Studio Render-Test")
    parser.add_argument("--audio", required=True, help="Pfad zur Audiodatei")
    parser.add_argument("--videos", required=True, help="Ordner mit Video-Clips")
    parser.add_argument("--output", default="test_output.mp4", help="Output-Datei")
    parser.add_argument("--duration", type=float, default=None, 
                        help="Maximale Dauer in Sekunden (default: volle Länge)")
    parser.add_argument("--no-gpu", action="store_true", help="GPU deaktivieren")
    
    args = parser.parse_args()
    
    success = run_full_render_test(
        audio_path=args.audio,
        video_dir=args.videos,
        output_path=args.output,
        duration_limit=args.duration,
        use_gpu=not args.no_gpu
    )
    
    sys.exit(0 if success else 1)
