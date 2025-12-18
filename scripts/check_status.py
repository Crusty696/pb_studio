#!/usr/bin/env python3
"""
STATUS CHECKER - ASCII Only
"""

import json
import time
from pathlib import Path


def check_status():
    print("STATUS CHECK - Intelligentes Pacing System")
    print("=" * 50)

    # 1. Musik-Analyse Status
    analysis_file = Path("intelligent_pacing_schlaflos_studio54.json")

    if analysis_file.exists():
        try:
            with open(analysis_file, encoding="utf-8") as f:
                data = json.load(f)

            stats = data.get("statistics", {})
            print("MUSIK-ANALYSE: ABGESCHLOSSEN")
            print(f"Audio-Dauer: {stats.get('total_duration', 0)/60:.1f} min")
            print(f"BPM: {stats.get('bpm', 0):.1f}")
            print(f"Cuts total: {stats.get('total_cuts', 0)}")
            print(f"Avg Clip-Laenge: {stats.get('avg_clip_duration', 0):.2f}s")

            energy_dist = stats.get("energy_distribution", {})
            print("\nEnergy-Verteilung:")
            for level, count in energy_dist.items():
                percentage = (count / stats.get("total_cuts", 1)) * 100
                print(f"  {level.upper()}: {count} ({percentage:.1f}%)")

            print("\nBereit fuer intelligentes Video-Rendering!")
            return True

        except Exception as e:
            print(f"MUSIK-ANALYSE: FEHLER - {e}")
            return False
    else:
        print("MUSIK-ANALYSE: LAEUFT...")
        print("Analysiert: BPM, Beats, Energy-Kurve, Onsets")
        print("Geschätzte Zeit: 5-10 Minuten")
        return False


if __name__ == "__main__":
    check_status()
