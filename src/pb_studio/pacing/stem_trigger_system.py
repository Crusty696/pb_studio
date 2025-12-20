"""
Stem-basierte Trigger-Analyse fuer praeziseres Pacing.

Analysiert separierte Audio-Stems (Drums, Bass, Vocals) direkt,
ohne Frequenzfilterung. Ergibt praezisere Kick/Snare/HiHat-Erkennung
besonders bei DJ-Mixes mit starker Instrumenten-Ueberlappung.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StemTriggers:
    """Container fuer Trigger-Zeiten aus einem Stem."""

    stem_name: str
    times: np.ndarray  # Trigger-Zeitpunkte in Sekunden
    strengths: np.ndarray  # Trigger-Staerken (0-1)
    trigger_type: str  # z.B. "kick", "snare", "bass_drop"


@dataclass
class DrumsTriggers:
    """Drums-spezifische Trigger."""

    kick_times: np.ndarray = field(default_factory=lambda: np.array([]))
    kick_strengths: np.ndarray = field(default_factory=lambda: np.array([]))
    snare_times: np.ndarray = field(default_factory=lambda: np.array([]))
    snare_strengths: np.ndarray = field(default_factory=lambda: np.array([]))
    hihat_times: np.ndarray = field(default_factory=lambda: np.array([]))
    hihat_strengths: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class BassTriggers:
    """Bass-spezifische Trigger."""

    bass_drop_times: np.ndarray = field(default_factory=lambda: np.array([]))
    bass_drop_strengths: np.ndarray = field(default_factory=lambda: np.array([]))
    sub_energy_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    sub_energy_times: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class MelodyTriggers:
    """Melody/Synth-spezifische Trigger (Other-Stem).

    Fuer EDM/DJ-Mixes: Erkennt melodische Hooks, Synth-Stabs und Leads.
    Da EDM wenig Vocals hat, nutzen wir den Other-Stem fuer Melodie-Erkennung.
    """

    # Melodie-Presence (kontinuierlich)
    melody_presence_times: np.ndarray = field(default_factory=lambda: np.array([]))
    melody_presence_values: np.ndarray = field(default_factory=lambda: np.array([]))
    # Synth-Stabs/Hooks (diskret)
    synth_hit_times: np.ndarray = field(default_factory=lambda: np.array([]))
    synth_hit_strengths: np.ndarray = field(default_factory=lambda: np.array([]))
    # Melodie-Hooks (markante Stellen)
    hook_times: np.ndarray = field(default_factory=lambda: np.array([]))
    hook_strengths: np.ndarray = field(default_factory=lambda: np.array([]))


class DrumsTriggerAnalyzer:
    """
    Analysiert Drums-Stem fuer Kick, Snare und HiHat.

    Direkte Onset-Detection auf isoliertem Drums-Signal.
    Keine Frequenzfilterung noetig - Signal enthaelt nur Drums!
    """

    def __init__(self, sr: int = 22050, hop_length: int = 512):
        """
        Args:
            sr: Sample Rate fuer Analyse
            hop_length: Hop Length fuer Onset-Detection
        """
        self.sr = sr
        self.hop_length = hop_length

    def analyze(
        self, drums_path: Path, progress_callback: Callable[[float], None] | None = None
    ) -> DrumsTriggers:
        """
        Analysiert Drums-Stem.

        Args:
            drums_path: Pfad zum Drums-Stem WAV
            progress_callback: Optional callback(progress_0_to_1)

        Returns:
            DrumsTriggers mit Kick/Snare/HiHat Zeiten
        """
        logger.info(f"Analysiere Drums-Stem: {drums_path}")

        # Audio laden
        y, sr = librosa.load(drums_path, sr=self.sr, mono=True)

        if progress_callback:
            progress_callback(0.2)

        # Onset-Envelope berechnen
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)

        # Onsets detektieren
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=self.hop_length, backtrack=True
        )
        onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=self.hop_length)

        if progress_callback:
            progress_callback(0.4)

        # Frequenzbasierte Klassifikation der Onsets
        # Berechne Spektrum um jeden Onset
        kick_times, kick_strengths = [], []
        snare_times, snare_strengths = [], []
        hihat_times, hihat_strengths = [], []

        # Fuer jeden Onset: Analysiere lokales Spektrum
        for onset_time in onset_times:
            # Sample-Position
            sample_pos = int(onset_time * sr)
            window_size = int(0.05 * sr)  # 50ms Fenster

            # Extrahiere Fenster um Onset
            start = max(0, sample_pos - window_size // 4)
            end = min(len(y), sample_pos + window_size)
            window = y[start:end]

            if len(window) < 256:
                continue

            # Spektrum berechnen
            spectrum = np.abs(np.fft.rfft(window))
            freqs = np.fft.rfftfreq(len(window), 1 / sr)

            # Energie in Frequenzbaendern
            low_mask = freqs < 200  # Kick: < 200 Hz
            mid_mask = (freqs >= 200) & (freqs < 2000)  # Snare: 200-2000 Hz
            high_mask = freqs >= 5000  # HiHat: > 5000 Hz

            low_energy = np.sum(spectrum[low_mask]) if np.any(low_mask) else 0
            mid_energy = np.sum(spectrum[mid_mask]) if np.any(mid_mask) else 0
            high_energy = np.sum(spectrum[high_mask]) if np.any(high_mask) else 0

            total_energy = low_energy + mid_energy + high_energy + 1e-10

            # Klassifikation basierend auf dominanter Energie
            low_ratio = low_energy / total_energy
            mid_ratio = mid_energy / total_energy
            high_ratio = high_energy / total_energy

            # Onset-Staerke aus Envelope
            onset_frame = int(onset_time * sr / self.hop_length)
            if onset_frame < len(onset_env):
                strength = onset_env[onset_frame] / (np.max(onset_env) + 1e-10)
            else:
                strength = 0.5

            # Klassifiziere
            if low_ratio > 0.4:  # Dominant tieffrequent
                kick_times.append(onset_time)
                kick_strengths.append(strength)
            elif mid_ratio > 0.4:  # Dominant mittelfrequent
                snare_times.append(onset_time)
                snare_strengths.append(strength)
            elif high_ratio > 0.3:  # Dominant hochfrequent
                hihat_times.append(onset_time)
                hihat_strengths.append(strength)
            else:
                # Unklassifiziert - nehme staerkste
                if low_ratio >= mid_ratio and low_ratio >= high_ratio:
                    kick_times.append(onset_time)
                    kick_strengths.append(strength * 0.5)
                elif mid_ratio >= high_ratio:
                    snare_times.append(onset_time)
                    snare_strengths.append(strength * 0.5)
                else:
                    hihat_times.append(onset_time)
                    hihat_strengths.append(strength * 0.5)

        if progress_callback:
            progress_callback(1.0)

        logger.info(
            f"Drums-Analyse: {len(kick_times)} Kicks, "
            f"{len(snare_times)} Snares, {len(hihat_times)} HiHats"
        )

        return DrumsTriggers(
            kick_times=np.array(kick_times),
            kick_strengths=np.array(kick_strengths),
            snare_times=np.array(snare_times),
            snare_strengths=np.array(snare_strengths),
            hihat_times=np.array(hihat_times),
            hihat_strengths=np.array(hihat_strengths),
        )


class BassTriggerAnalyzer:
    """
    Analysiert Bass-Stem fuer Bass-Drops und Sub-Bass Energie.
    """

    def __init__(self, sr: int = 22050, hop_length: int = 512):
        self.sr = sr
        self.hop_length = hop_length

    def analyze(
        self, bass_path: Path, progress_callback: Callable[[float], None] | None = None
    ) -> BassTriggers:
        """
        Analysiert Bass-Stem.

        Args:
            bass_path: Pfad zum Bass-Stem WAV
            progress_callback: Optional callback(progress_0_to_1)

        Returns:
            BassTriggers mit Bass-Drop Zeiten und Sub-Bass Energie
        """
        logger.info(f"Analysiere Bass-Stem: {bass_path}")

        # Audio laden
        y, sr = librosa.load(bass_path, sr=self.sr, mono=True)

        if progress_callback:
            progress_callback(0.2)

        # RMS Energie berechnen
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=self.hop_length)

        if progress_callback:
            progress_callback(0.5)

        # Bass-Drops: Schnelle Energie-Anstiege
        # Berechne Differenz der RMS
        rms_diff = np.diff(rms, prepend=rms[0])

        # Threshold fuer Drops (95. Perzentil der positiven Differenzen)
        positive_diffs = rms_diff[rms_diff > 0]
        if len(positive_diffs) > 0:
            drop_threshold = np.percentile(positive_diffs, 95)
        else:
            drop_threshold = 0.1

        # Finde Drop-Positionen
        drop_frames = np.where(rms_diff > drop_threshold)[0]

        # Merge nahegelegene Drops (innerhalb 200ms)
        min_distance_frames = int(0.2 * sr / self.hop_length)
        merged_drops = []
        last_drop = -min_distance_frames

        for frame in drop_frames:
            if frame - last_drop >= min_distance_frames:
                merged_drops.append(frame)
                last_drop = frame

        drop_times = librosa.frames_to_time(merged_drops, sr=sr, hop_length=self.hop_length)
        drop_strengths = (
            rms_diff[merged_drops] / (np.max(rms_diff) + 1e-10) if merged_drops else np.array([])
        )

        if progress_callback:
            progress_callback(0.8)

        # Sub-Bass Energie (Low-Pass gefiltert)
        # Da wir schon den isolierten Bass haben, ist RMS ein guter Proxy
        sub_energy = rms / (np.max(rms) + 1e-10)

        if progress_callback:
            progress_callback(1.0)

        logger.info(f"Bass-Analyse: {len(drop_times)} Bass-Drops erkannt")

        return BassTriggers(
            bass_drop_times=np.array(drop_times),
            bass_drop_strengths=drop_strengths,
            sub_energy_curve=sub_energy,
            sub_energy_times=rms_times,
        )


class MelodyTriggerAnalyzer:
    """
    Analysiert Other-Stem fuer Melodien, Synth-Stabs und Hooks.

    Fuer EDM/DJ-Mixes: Der "Other"-Stem enthaelt Melodien, Leads, Synths.
    Erkennt markante Stellen fuer effektvolle Schnitte.
    """

    def __init__(self, sr: int = 22050, hop_length: int = 512):
        self.sr = sr
        self.hop_length = hop_length

    def analyze(
        self, other_path: Path, progress_callback: Callable[[float], None] | None = None
    ) -> MelodyTriggers:
        """
        Analysiert Other-Stem (Melodien/Synths).

        Args:
            other_path: Pfad zum Other-Stem WAV
            progress_callback: Optional callback(progress_0_to_1)

        Returns:
            MelodyTriggers mit Presence, Synth-Hits und Hooks
        """
        logger.info(f"Analysiere Melody/Synth-Stem: {other_path}")

        # Audio laden
        y, sr = librosa.load(other_path, sr=self.sr, mono=True)

        if progress_callback:
            progress_callback(0.15)

        # RMS fuer Melody-Presence
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=self.hop_length)

        # Normalisiere Presence (0-1)
        rms_max = np.max(rms) + 1e-10
        melody_presence = rms / rms_max

        if progress_callback:
            progress_callback(0.35)

        # Onset-Detection fuer Synth-Stabs
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=self.hop_length, backtrack=True
        )
        onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=self.hop_length)

        # Onset-Staerken
        onset_strengths = []
        for onset_frame in onsets:
            if onset_frame < len(onset_env):
                strength = onset_env[onset_frame] / (np.max(onset_env) + 1e-10)
            else:
                strength = 0.5
            onset_strengths.append(strength)

        if progress_callback:
            progress_callback(0.55)

        # Synth-Stabs: Starke, kurze Onsets (Strength > 0.6)
        stab_mask = np.array(onset_strengths) > 0.6
        synth_hit_times = onset_times[stab_mask] if np.any(stab_mask) else np.array([])
        synth_hit_strengths = (
            np.array(onset_strengths)[stab_mask] if np.any(stab_mask) else np.array([])
        )

        if progress_callback:
            progress_callback(0.70)

        # Hook-Detection: Melodische Highlights (lokale Maxima in Energie)
        # Glaette RMS fuer stabiler Hook-Detection
        window_size = int(2.0 * sr / self.hop_length)  # 2 Sekunden Fenster
        if window_size > 1 and len(melody_presence) > window_size:
            kernel = np.ones(window_size) / window_size
            smoothed = np.convolve(melody_presence, kernel, mode="same")
        else:
            smoothed = melody_presence

        # Finde lokale Maxima (Hooks)
        from scipy.signal import find_peaks

        # Mindestabstand: 4 Sekunden zwischen Hooks
        min_distance = int(4.0 * sr / self.hop_length)
        peaks, properties = find_peaks(
            smoothed,
            distance=min_distance,
            prominence=0.15,  # Mindest-Prominence
        )

        hook_times = rms_times[peaks] if len(peaks) > 0 else np.array([])
        hook_strengths = smoothed[peaks] if len(peaks) > 0 else np.array([])

        if progress_callback:
            progress_callback(1.0)

        # Statistik
        active_ratio = np.sum(melody_presence > 0.15) / len(melody_presence)
        logger.info(
            f"Melody-Analyse: {active_ratio * 100:.1f}% aktiv, "
            f"{len(synth_hit_times)} Synth-Stabs, {len(hook_times)} Hooks"
        )

        return MelodyTriggers(
            melody_presence_times=rms_times,
            melody_presence_values=melody_presence,
            synth_hit_times=synth_hit_times,
            synth_hit_strengths=synth_hit_strengths,
            hook_times=hook_times,
            hook_strengths=hook_strengths,
        )


class StemTriggerMerger:
    """
    Kombiniert Trigger aus allen Stems zu einem einheitlichen Trigger-Set.

    Fuer EDM/DJ-Mixes optimiert: Nutzt Melody statt Vocals.
    """

    def __init__(self, hook_boost_factor: float = 1.5, min_trigger_distance: float = 0.05):
        """
        Args:
            hook_boost_factor: Faktor um Hook-Trigger zu verstaerken (1.0-2.0)
            min_trigger_distance: Minimaler Abstand zwischen Triggern in Sekunden
        """
        self.hook_boost_factor = hook_boost_factor
        self.min_trigger_distance = min_trigger_distance

    def merge(
        self,
        drums: DrumsTriggers | None = None,
        bass: BassTriggers | None = None,
        melody: MelodyTriggers | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Kombiniert alle Stem-Trigger.

        Args:
            drums: DrumsTriggers oder None
            bass: BassTriggers oder None
            melody: MelodyTriggers oder None (fuer Synth-Stabs und Hooks)

        Returns:
            Dict mit Trigger-Typ -> (Zeiten, Staerken):
            {
                "kick": (times, strengths),
                "snare": (times, strengths),
                "hihat": (times, strengths),
                "bass_drop": (times, strengths),
                "synth_stab": (times, strengths),  # Synth-Akzente
                "melody_hook": (times, strengths)  # Melodie-Highlights
            }
        """
        result = {}

        # Drums-Trigger
        if drums is not None:
            if len(drums.kick_times) > 0:
                result["kick"] = (drums.kick_times, drums.kick_strengths)
            if len(drums.snare_times) > 0:
                result["snare"] = (drums.snare_times, drums.snare_strengths)
            if len(drums.hihat_times) > 0:
                result["hihat"] = (drums.hihat_times, drums.hihat_strengths)

        # Bass-Trigger
        if bass is not None and len(bass.bass_drop_times) > 0:
            result["bass_drop"] = (bass.bass_drop_times, bass.bass_drop_strengths)

        # Melody-Trigger (Synth-Stabs und Hooks)
        if melody is not None:
            # Synth-Stabs als Cut-Trigger
            if len(melody.synth_hit_times) > 0:
                result["synth_stab"] = (melody.synth_hit_times, melody.synth_hit_strengths)

            # Melodie-Hooks (verstaerkt fuer dramatische Schnitte)
            if len(melody.hook_times) > 0:
                boosted_strengths = np.clip(
                    melody.hook_strengths * self.hook_boost_factor, 0.0, 1.0
                )
                result["melody_hook"] = (melody.hook_times, boosted_strengths)

        logger.info(f"StemTriggerMerger: {len(result)} Trigger-Typen zusammengefuehrt")
        for trigger_type, (times, _) in result.items():
            logger.debug(f"  - {trigger_type}: {len(times)} Trigger")

        return result

    def boost_triggers_at_hooks(
        self,
        trigger_times: np.ndarray,
        trigger_strengths: np.ndarray,
        melody: MelodyTriggers,
        boost_radius_sec: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Verstaerkt Trigger die in der Naehe von Melodie-Hooks liegen.

        Fuer EDM: Cuts sollen an melodischen Highlights staerker sein.

        Args:
            trigger_times: Original Trigger-Zeiten
            trigger_strengths: Original Trigger-Staerken
            melody: MelodyTriggers mit Hook-Zeiten
            boost_radius_sec: Radius um Hooks in Sekunden

        Returns:
            (zeiten, verstaerkte_staerken)
        """
        if len(melody.hook_times) == 0:
            return trigger_times, trigger_strengths

        boosted_strengths = trigger_strengths.copy()

        for i, t in enumerate(trigger_times):
            # Finde naechsten Hook
            distances = np.abs(melody.hook_times - t)
            min_dist = np.min(distances)

            if min_dist < boost_radius_sec:
                # Boost proportional zur Naehe
                boost = 1.0 + (self.hook_boost_factor - 1.0) * (1.0 - min_dist / boost_radius_sec)
                boosted_strengths[i] = np.clip(boosted_strengths[i] * boost, 0.0, 1.0)

        return trigger_times, boosted_strengths
