"""
Clip Calculation Logic
Extracted from CutListController to separate concerns.
"""
import random


def calculate_intelligent_clip_segment(
    clip_duration: float,
    segment_duration: float,
    trigger_strength: float = 0.5,
    previous_clip_end: float = 0.0,
    clip_id: str = "",
) -> tuple[float, float]:
    """
    Berechnet intelligenten Startpunkt und Dauer für einen Clip-Ausschnitt.

    Strategie:
    1. Wenn Clip kürzer als Segment → ganzes Video zeigen
    2. Bei hoher Trigger-Stärke → längere Clips (mehr vom Video zeigen)
    3. Startpunkt variiert, um verschiedene Teile des Videos zu zeigen
    4. Vermeidet die letzten 0.5s (oft schwarzer Fade-Out)

    Args:
        clip_duration: Gesamtdauer des Video-Clips in Sekunden
        segment_duration: Gewünschte Segment-Dauer (aus Trigger-Abstand)
        trigger_strength: Stärke des Triggers (0.0-1.0), beeinflusst Clip-Länge
        previous_clip_end: Ende des vorherigen Clips (für Variation)
        clip_id: Clip-ID für konsistente Randomisierung

    Returns:
        (clip_start, actual_duration) - Startpunkt im Clip und tatsächliche Dauer
    """
    # Sicherheitsmargin am Ende des Clips (Fade-Out vermeiden)
    SAFETY_MARGIN = 0.3
    usable_duration = max(0.5, clip_duration - SAFETY_MARGIN)

    # Minimum und Maximum Clip-Dauer
    MIN_CLIP_DURATION = 1.5  # Mindestens 1.5 Sekunden zeigen
    MAX_CLIP_DURATION = min(15.0, usable_duration)  # Maximal 15s oder ganze Clip-Länge

    # Wenn Clip kürzer als gewünschte Segment-Dauer → ganzen Clip zeigen
    if usable_duration <= segment_duration:
        return 0.0, usable_duration

    # Variable Clip-Dauer basierend auf Trigger-Stärke
    # Hohe Stärke (>0.7) = längere Clips (mehr zeigen)
    # Niedrige Stärke (<0.3) = kürzere Clips (schneller weiter)
    strength_factor = 0.5 + (trigger_strength * 0.5)  # 0.5 bis 1.0

    # Basis-Dauer mit Variation
    base_duration = segment_duration * strength_factor

    # Zufällige Variation hinzufügen (±20%)
    # Verwende clip_id für konsistente Randomisierung
    random.seed(hash(clip_id) % 10000)
    variation = random.uniform(0.8, 1.2)
    actual_duration = base_duration * variation

    # Clamp auf erlaubten Bereich
    actual_duration = max(MIN_CLIP_DURATION, min(MAX_CLIP_DURATION, actual_duration))

    # Wenn Clip lang genug → variablen Startpunkt wählen
    if usable_duration > actual_duration + 0.5:
        # Maximaler Startpunkt (lässt Platz für die Dauer)
        max_start = usable_duration - actual_duration

        # Wähle Startpunkt mit leichter Präferenz für den Anfang (wo meist Aktion ist)
        # Aber nicht immer 0.0!
        random.seed(hash(clip_id + str(segment_duration)) % 10000)

        # 60% Chance auf ersten Drittel, 30% auf Mitte, 10% auf Ende
        roll = random.random()
        if roll < 0.6:
            clip_start = random.uniform(0, max_start * 0.33)
        elif roll < 0.9:
            clip_start = random.uniform(max_start * 0.33, max_start * 0.66)
        else:
            clip_start = random.uniform(max_start * 0.66, max_start)

        clip_start = round(clip_start, 2)
    else:
        clip_start = 0.0

    return clip_start, round(actual_duration, 2)
