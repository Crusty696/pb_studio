"""
Dynamic Duration Calculator für PB_studio

Berechnet variable Clip-Längen basierend auf:
- Musikalischen Phrasen (4/8/16 Beats)
- Audio-Energy und Song-Struktur
- Trigger-Stärke (natürliche Schnitt-Punkte)
- Clip-Match-Quality (perfekte Matches dürfen länger laufen)

Ersetzt das alte "fester Abstand zwischen Triggers" System.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DurationConstraints:
    """
    Constraints für Clip-Dauer Berechnung.

    Attributes:
        min_duration: Minimale Clip-Dauer (s)
        max_duration: Maximale Clip-Dauer (s)
        phrase_beats: Bevorzugte Phrase-Länge in Beats (4, 8, 16)
        allow_full_clip: Erlaube volle Clip-Dauer bei perfektem Match
        max_full_clip: Maximale Dauer für volle Clips (s)
    """

    min_duration: float = 2.0
    max_duration: float = 16.0
    phrase_beats: int = 4  # 4/8/16 Beats
    allow_full_clip: bool = True
    max_full_clip: float = 20.0


class DynamicDurationCalculator:
    """
    Berechnet variable Clip-Längen für natürlichere Video-Edits.

    Ersetzt das alte fixed-duration System mit intelligentem Algorithmus:
    1. Basis-Dauer aus musikalischen Phrasen (4/8/16 Beats)
    2. Anpassung nach Song-Struktur (Intro länger, Chorus kürzer)
    3. Energy-basierte Modulation (hohe Energy = schneller)
    4. Full-Clip-Allowance bei perfekter Match-Quality
    """

    def __init__(self, bpm: float, constraints: DurationConstraints | None = None):
        """
        Initialisiert Duration Calculator.

        Args:
            bpm: BPM des Songs
            constraints: Optional DurationConstraints (verwendet Defaults)

        Raises:
            ValueError: If BPM is invalid (<= 0)
        """
        # CRITICAL FIX: Validate BPM to prevent division by zero
        if bpm is None or bpm <= 0:
            logger.warning(f"Invalid BPM={bpm}, using fallback BPM=120.0")
            bpm = 120.0

        # Additional sanity check: BPM should be in reasonable range (30-300)
        if not (30.0 <= bpm <= 300.0):
            logger.warning(f"BPM={bpm:.1f} outside normal range (30-300), " f"using clamped value")
            bpm = max(30.0, min(300.0, bpm))

        self.bpm = bpm
        self.beat_duration = 60.0 / bpm  # Safe now - BPM validated
        self.constraints = constraints or DurationConstraints()

        logger.info(
            f"DynamicDurationCalculator initialized: BPM={bpm:.1f}, "
            f"beat_duration={self.beat_duration:.3f}s, "
            f"phrase_beats={self.constraints.phrase_beats}"
        )

    def calculate_duration(
        self,
        audio_energy: float,
        trigger_strength: float,
        segment_type: str | None = None,
        match_quality: float = 0.5,
        clip_duration: float | None = None,
        next_trigger_distance: float | None = None,
    ) -> float:
        """
        Berechnet optimale Clip-Dauer.

        Args:
            audio_energy: Audio-Energy zum Zeitpunkt (0.0-1.0)
            trigger_strength: Stärke des Triggers (0.0-1.0)
            segment_type: Song-Segment ('intro', 'verse', 'chorus', 'drop', 'outro')
            match_quality: Match-Quality des Clips (0.0-1.0)
            clip_duration: Tatsächliche Clip-Dauer (None = unbekannt)
            next_trigger_distance: Abstand zum nächsten Trigger (s)

        Returns:
            Optimale Clip-Dauer in Sekunden
        """
        # 1. FULL CLIP ALLOWANCE: Perfekte Matches dürfen länger laufen
        if self.constraints.allow_full_clip and clip_duration is not None and match_quality >= 0.85:
            # Volle Clip-Dauer, aber begrenzt
            full_clip_duration = min(clip_duration, self.constraints.max_full_clip)

            # Nur wenn musikalisch sinnvoll (4-20s)
            if (
                self.constraints.min_duration
                <= full_clip_duration
                <= self.constraints.max_full_clip
            ):
                logger.debug(
                    f"Full clip allowed: match_quality={match_quality:.2f}, "
                    f"duration={full_clip_duration:.1f}s"
                )
                return full_clip_duration

        # 2. MUSIKALISCHE PHRASEN: Basis-Dauer aus Beats
        phrase_duration = self._get_phrase_duration(segment_type, audio_energy)

        # 3. ENERGY MODULATION: Hohe Energy = kürzere Clips
        energy_factor = self._get_energy_factor(audio_energy)
        modulated_duration = phrase_duration * energy_factor

        # 4. TRIGGER STRENGTH: Starke Trigger = natürliche Schnitt-Punkte
        if next_trigger_distance is not None and trigger_strength >= 0.7:
            # Snap zu nächstem starken Trigger wenn nah
            if next_trigger_distance < modulated_duration * 1.2:
                modulated_duration = next_trigger_distance

        # 5. CONSTRAINTS: Min/Max einhalten
        final_duration = max(
            self.constraints.min_duration, min(modulated_duration, self.constraints.max_duration)
        )

        logger.debug(
            f"Calculated duration: energy={audio_energy:.2f}, "
            f"phrase={phrase_duration:.1f}s, "
            f"modulated={modulated_duration:.1f}s, "
            f"final={final_duration:.1f}s"
        )

        return final_duration

    def _get_phrase_duration(self, segment_type: str | None, audio_energy: float) -> float:
        """
        Berechnet Phrase-Dauer basierend auf Song-Struktur.

        Args:
            segment_type: Song-Segment
            audio_energy: Audio-Energy

        Returns:
            Phrase-Dauer in Sekunden
        """
        # Basis: 4/8/16 Beats je nach Segment
        if segment_type == "intro" or segment_type == "outro":
            # Intro/Outro: Längere Phrasen (8-16 Beats)
            beats = 16 if audio_energy < 0.4 else 8
        elif segment_type == "drop" or segment_type == "chorus":
            # Drop/Chorus: Kürzere Phrasen (4 Beats)
            beats = 4
        elif segment_type == "verse":
            # Verse: Mittlere Phrasen (8 Beats)
            beats = 8
        elif segment_type == "bridge":
            # Bridge: Variable (4-8 Beats)
            beats = 8 if audio_energy < 0.5 else 4
        else:
            # Default: 4 Beats
            beats = self.constraints.phrase_beats

        duration = beats * self.beat_duration

        logger.debug(
            f"Phrase duration: segment={segment_type}, "
            f"beats={beats}, "
            f"duration={duration:.1f}s"
        )

        return duration

    def _get_energy_factor(self, audio_energy: float) -> float:
        """
        Berechnet Energy-Modulation-Faktor.

        Hohe Energy = schnellere Cuts (Factor < 1.0)
        Niedrige Energy = längere Clips (Factor > 1.0)

        Args:
            audio_energy: Audio-Energy (0.0-1.0)

        Returns:
            Modulation-Faktor (0.5-1.5)
        """
        if audio_energy > 0.8:
            # Sehr hohe Energy: 50% kürzer
            return 0.5
        elif audio_energy > 0.6:
            # Hohe Energy: 25% kürzer
            return 0.75
        elif audio_energy < 0.3:
            # Niedrige Energy: 50% länger
            return 1.5
        elif audio_energy < 0.5:
            # Mittlere Energy: 25% länger
            return 1.25
        else:
            # Normal: Keine Modulation
            return 1.0
