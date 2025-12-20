"""
Trigger-System für PB_studio Pacing Engine

Analysiert Audio-Dateien und extrahiert verschiedene Trigger-Typen:
- Beat: Regelmäßige Beats (BPM-Grid)
- Onset: Alle Transienten/Hits
- Kick: Bass-Drum Hits (20-200 Hz)
- Snare: Snare-Drum Hits (200-2000 Hz)
- HiHat: Hi-Hat Hits (2000+ Hz)
- Energy: Energiespitzen (RMS-basiert)

Basiert auf librosa und folgt Audio-Analyse-Best-Practices.

PERFORMANCE OPTIMIZATION: Trigger-Analyse-Caching
- Trigger-Analysen werden persistent gecacht
- Cache-Key basiert auf Datei-Hash (MD5)
- Typischer Speedup: 10-20x für gecachte Songs

STEM-BASED ANALYSIS (2025-12-08):
- Optional: Audio in Stems trennen (Drums, Bass, Vocals)
- Präzisere Kick/Snare/HiHat-Erkennung bei DJ-Mixes
- Nutzt audio-separator mit DirectML (AMD GPU)
"""

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# Stem-basierte Analyse (optional)
# Stem-basierte Analyse (optional)
try:
    from pb_studio.audio.stem_separator import AudioTooLongError
    from pb_studio.audio.stem_separator import StemSeparator as AudioSeparatorStemExtractor
    from pb_studio.pacing.stem_trigger_system import (
        BassTriggerAnalyzer,
        BassTriggers,
        DrumsTriggerAnalyzer,
        DrumsTriggers,
        MelodyTriggerAnalyzer,
        MelodyTriggers,
        StemTriggerMerger,
    )

    STEM_ANALYSIS_AVAILABLE = True
except ImportError:
    STEM_ANALYSIS_AVAILABLE = False
    AudioTooLongError = None  # Fallback fuer Import
    logger.debug("Stem-basierte Analyse nicht verfuegbar (audio-separator nicht installiert)")


@dataclass
class TriggerAnalysisResult:
    """
    Ergebnis einer Trigger-Analyse.

    Attributes:
        beat_times: Zeiten der Beat-Trigger (BPM-Grid) in Sekunden
        onset_times: Zeiten aller Onset-Trigger in Sekunden
        kick_times: Zeiten der Kick-Drum-Trigger in Sekunden
        snare_times: Zeiten der Snare-Drum-Trigger in Sekunden
        hihat_times: Zeiten der Hi-Hat-Trigger in Sekunden
        energy_times: Zeiten der Energie-Peaks in Sekunden
        bpm: Erkannte BPM (Beats per Minute)
        duration: Audio-Dauer in Sekunden
        sample_rate: Sample-Rate des analysierten Audios
    """

    beat_times: list[float]
    onset_times: list[float]
    kick_times: list[float]
    snare_times: list[float]
    hihat_times: list[float]
    energy_times: list[float]
    bpm: float
    duration: float
    sample_rate: int


class TriggerSystem:
    """
    Analysiert Audio-Dateien und extrahiert Trigger-Zeitpunkte.

    Verwendet librosa für:
    - Beat-Tracking (BPM-Grid)
    - Onset-Detection (Transienten)
    - Frequency-Band-Filtering (Kick/Snare/HiHat)
    - RMS-Energy-Analysis (Energiespitzen)
    """

    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 512,
        kick_freq_range: tuple = (20, 200),
        snare_freq_range: tuple = (200, 2000),
        hihat_freq_range: tuple = (2000, 20000),
        energy_percentile: float = 90.0,
        use_cache: bool = True,
        cache_dir: str | Path = "trigger_cache",
        use_stems: bool = False,
        stem_cache_dir: str | Path = "stem_cache",
    ):
        """
        Initialisiert das Trigger-System.

        Args:
            sr: Sample-Rate für Audio-Analyse
            hop_length: Hop-Length für STFT (kleinere Werte = präziser aber langsamer)
            kick_freq_range: Frequenzbereich für Kick-Detection in Hz
            snare_freq_range: Frequenzbereich für Snare-Detection in Hz
            hihat_freq_range: Frequenzbereich für HiHat-Detection in Hz
            energy_percentile: Perzentil für Energy-Peak-Schwellwert (90.0 = Top 10%)
            use_cache: Enable/disable trigger analysis caching (PERFORMANCE OPTIMIZATION)
            cache_dir: Directory for cached trigger analyses (PERFORMANCE OPTIMIZATION)
            use_stems: Enable stem-based analysis for better accuracy (DJ-Mixes)
            stem_cache_dir: Directory for cached audio stems
        """
        self.sr = sr
        self.hop_length = hop_length
        self.kick_freq_range = kick_freq_range
        self.snare_freq_range = snare_freq_range
        self.hihat_freq_range = hihat_freq_range
        self.energy_percentile = energy_percentile

        # PERFORMANCE OPTIMIZATION: Trigger analysis caching
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        if use_cache:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Trigger-Analyse-Cache aktiviert: {self.cache_dir}")

        # STEM-BASED ANALYSIS (2025-12-08)
        self.use_stems = use_stems and STEM_ANALYSIS_AVAILABLE
        self.stem_cache_dir = Path(stem_cache_dir)
        self._stem_extractor = None
        self._drums_analyzer = None
        self._bass_analyzer = None
        self._melody_analyzer = None  # Other-Stem fuer Melody/Synth
        self._stem_merger = None

        if use_stems:
            if STEM_ANALYSIS_AVAILABLE:
                logger.info("Stem-basierte Analyse aktiviert (praezisere Kick/Snare/HiHat)")
            else:
                logger.warning(
                    "Stem-basierte Analyse angefordert aber nicht verfuegbar. "
                    "Installiere audio-separator: pip install audio-separator[cpu]"
                )

        logger.info(
            f"TriggerSystem initialisiert: sr={sr}, hop_length={hop_length}, "
            f"kick={kick_freq_range}, snare={snare_freq_range}, hihat={hihat_freq_range}, "
            f"cache={'enabled' if use_cache else 'disabled'}, "
            f"stems={'enabled' if self.use_stems else 'disabled'}"
        )

    def _get_cache_key(
        self, audio_path: Path, start_time: float | None = None, end_time: float | None = None
    ) -> str:
        """
        Generate cache key from file path and metadata.

        PERFORMANCE OPTIMIZATION: MD5-based cache key ensures unique identification.

        Args:
            audio_path: Path to audio file
            start_time: Optional start time for segment caching
            end_time: Optional end time for segment caching

        Returns:
            MD5 hash for cache key
        """
        # BUG-04 FIX: Use SHA-256 instead of MD5 to reduce collision risk
        # MD5 has known collision vulnerabilities; SHA-256 is more robust
        hasher = hashlib.sha256()
        hasher.update(str(audio_path.resolve()).encode())  # Use absolute path for consistency
        stat = audio_path.stat()
        # Include file size and modification time for uniqueness
        hasher.update(f"{stat.st_size}_{stat.st_mtime}".encode())

        # Add segment info if provided (for time-windowed caching)
        if start_time is not None and end_time is not None:
            hasher.update(f"_segment_{start_time:.2f}_{end_time:.2f}".encode())

        return hasher.hexdigest()

    def _load_trigger_cache(self, cache_key: str) -> TriggerAnalysisResult | None:
        """
        Load cached trigger analysis if available.

        PERFORMANCE OPTIMIZATION: Avoids expensive re-analysis of same audio.

        Args:
            cache_key: Cache key from _get_cache_key()

        Returns:
            Cached TriggerAnalysisResult or None if not found
        """
        if not self.use_cache:
            return None

        cache_file = self.cache_dir / f"{cache_key}_triggers.json"
        if cache_file.exists():
            try:
                with open(cache_file, encoding="utf-8") as f:
                    data = json.load(f)

                # Reconstruct TriggerAnalysisResult from JSON
                result = TriggerAnalysisResult(
                    beat_times=data["beat_times"],
                    onset_times=data["onset_times"],
                    kick_times=data["kick_times"],
                    snare_times=data["snare_times"],
                    hihat_times=data["hihat_times"],
                    energy_times=data["energy_times"],
                    bpm=data["bpm"],
                    duration=data["duration"],
                    sample_rate=data["sample_rate"],
                )

                logger.info(
                    f"✅ Trigger-Analyse aus Cache geladen (10-20x schneller!): {cache_key[:8]}..."
                )
                return result

            except Exception as e:
                logger.warning(f"Failed to load trigger cache: {e}")
                return None

        return None

    def _save_trigger_cache(self, cache_key: str, result: TriggerAnalysisResult) -> None:
        """
        Save trigger analysis to cache.

        PERFORMANCE OPTIMIZATION: Stores analysis for future reuse.

        Args:
            cache_key: Cache key from _get_cache_key()
            result: TriggerAnalysisResult to cache
        """
        if not self.use_cache:
            return

        cache_file = self.cache_dir / f"{cache_key}_triggers.json"
        try:
            # Convert dataclass to dict for JSON serialization
            data = asdict(result)

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Trigger-Analyse in Cache gespeichert: {cache_key[:8]}...")

        except Exception as e:
            logger.warning(f"Failed to save trigger cache: {e}")

    def analyze_triggers(
        self,
        audio_path: str,
        expected_bpm: float | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> TriggerAnalysisResult:
        """
        Analysiert Audio-Datei und extrahiert alle Trigger.

        Args:
            audio_path: Pfad zur Audio-Datei
            expected_bpm: Erwartete BPM (optional, verbessert Beat-Tracking)
            start_time: Optional start time in seconds (analyzes only this segment)
            end_time: Optional end time in seconds (analyzes only this segment)

        Returns:
            TriggerAnalysisResult mit allen extrahierten Triggern

        Raises:
            FileNotFoundError: Wenn Audio-Datei nicht existiert
            librosa.util.exceptions.ParameterError: Bei ungültigen Audio-Parametern
        """
        # Validierung
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio-Datei nicht gefunden: {audio_path}")

        # PERFORMANCE OPTIMIZATION: Check cache first (10-20x faster!)
        cache_key = self._get_cache_key(audio_file, start_time, end_time)
        cached_result = self._load_trigger_cache(cache_key)
        if cached_result is not None:
            return cached_result

        if start_time is not None and end_time is not None:
            logger.info(
                f"Starte Trigger-Analyse: {audio_path} (Zeitfenster: {start_time:.1f}s - {end_time:.1f}s)"
            )
        else:
            logger.info(f"Starte Trigger-Analyse: {audio_path}")

        # Audio laden (nur Zeitfenster wenn angegeben)
        if start_time is not None and end_time is not None:
            # PERFORMANCE OPTIMIZATION: Nur benötigtes Segment laden
            segment_duration = end_time - start_time
            y, sr = librosa.load(
                audio_path, sr=self.sr, offset=start_time, duration=segment_duration
            )
            duration = segment_duration
            logger.debug(
                f"Audio-Segment geladen: offset={start_time:.1f}s, duration={duration:.2f}s, sr={sr}"
            )
        else:
            # Gesamte Audio laden
            y, sr = librosa.load(audio_path, sr=self.sr)
            duration = librosa.get_duration(y=y, sr=sr)
            logger.debug(f"Audio geladen: duration={duration:.2f}s, sr={sr}")

        # Beat-Tracking (BPM-Grid)
        beat_times, bpm = self._detect_beats(y, sr, expected_bpm)
        logger.debug(f"Beats erkannt: {len(beat_times)} beats @ {bpm:.1f} BPM")

        # Onset-Detection (alle Transienten)
        onset_times = self._detect_onsets(y, sr)
        logger.debug(f"Onsets erkannt: {len(onset_times)} onsets")

        # Frequency-Band Onset Detection
        kick_times = self._detect_kick_onsets(y, sr)
        logger.debug(f"Kicks erkannt: {len(kick_times)} kicks")

        snare_times = self._detect_snare_onsets(y, sr)
        logger.debug(f"Snares erkannt: {len(snare_times)} snares")

        hihat_times = self._detect_hihat_onsets(y, sr)
        logger.debug(f"HiHats erkannt: {len(hihat_times)} hihats")

        # Energy-Peak Detection
        energy_times = self._detect_energy_peaks(y, sr)
        logger.debug(f"Energy-Peaks erkannt: {len(energy_times)} peaks")

        # IMPORTANT: Wenn Zeitfenster verwendet wurde, Trigger-Zeiten korrigieren
        # librosa gibt relative Zeiten (0-90s) zurück, wir brauchen absolute Zeiten
        if start_time is not None:
            logger.debug(f"Korrigiere Trigger-Zeiten um Offset: +{start_time:.1f}s")
            beat_times = [t + start_time for t in beat_times]
            onset_times = [t + start_time for t in onset_times]
            kick_times = [t + start_time for t in kick_times]
            snare_times = [t + start_time for t in snare_times]
            hihat_times = [t + start_time for t in hihat_times]
            energy_times = [t + start_time for t in energy_times]

        logger.info(
            f"Trigger-Analyse abgeschlossen: {len(beat_times)} beats, "
            f"{len(onset_times)} onsets, {len(kick_times)} kicks, "
            f"{len(snare_times)} snares, {len(hihat_times)} hihats, "
            f"{len(energy_times)} energy peaks"
        )

        # Create result object
        result = TriggerAnalysisResult(
            beat_times=beat_times,
            onset_times=onset_times,
            kick_times=kick_times,
            snare_times=snare_times,
            hihat_times=hihat_times,
            energy_times=energy_times,
            bpm=bpm,
            duration=duration,
            sample_rate=sr,
        )

        # PERFORMANCE OPTIMIZATION: Save to cache for future reuse
        self._save_trigger_cache(cache_key, result)

        return result

    def _detect_beats(
        self, y: np.ndarray, sr: int, expected_bpm: float | None = None
    ) -> tuple[list[float], float]:
        """
        Erkennt Beat-Grid (regelmäßige Beats).

        Args:
            y: Audio-Zeitreihe
            sr: Sample-Rate
            expected_bpm: Erwartete BPM (optional)

        Returns:
            Tuple (beat_times, bpm)
        """
        # Onset-Envelope für Beat-Tracking berechnen
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)

        # Tempo (BPM) schätzen
        if expected_bpm:
            # Mit Prior-BPM (präziser)
            tempo, beats = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=sr,
                hop_length=self.hop_length,
                start_bpm=expected_bpm,
                tightness=100,  # Hohe Tightness = bleibt nah am expected_bpm
            )
        else:
            # Ohne Prior-BPM
            tempo, beats = librosa.beat.beat_track(
                onset_envelope=onset_env, sr=sr, hop_length=self.hop_length
            )

        # Beats zu Zeitpunkten konvertieren
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
        bpm = float(tempo)

        return beat_times.tolist(), bpm

    def _detect_onsets(self, y: np.ndarray, sr: int) -> list[float]:
        """
        Erkennt alle Onsets (Transienten).

        Args:
            y: Audio-Zeitreihe
            sr: Sample-Rate

        Returns:
            Liste von Onset-Zeiten in Sekunden
        """
        # Onset-Frames erkennen
        onset_frames = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            hop_length=self.hop_length,
            backtrack=True,  # Präzisere Onset-Lokalisierung
        )

        # Zu Zeitpunkten konvertieren
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)

        return onset_times.tolist()

    def _detect_kick_onsets(self, y: np.ndarray, sr: int) -> list[float]:
        """
        Erkennt Kick-Drum Onsets mit präziser Frequenzband-Analyse.

        VERBESSERT: Nutzt Onset-Strength mit Mel-Spektrogram statt einfachem Bandpass.
        Frequenzbereich: 20-150Hz für präzise Kick-Erkennung.

        Args:
            y: Audio-Zeitreihe
            sr: Sample-Rate

        Returns:
            Liste von Kick-Zeiten in Sekunden
        """
        try:
            # Präzise Onset-Strength im Kick-Frequenzbereich
            kick_onset_env = librosa.onset.onset_strength(
                y=y,
                sr=sr,
                hop_length=self.hop_length,
                aggregate=np.median,
                fmin=20,
                fmax=150,
                n_mels=32,
            )

            # Peak-Picking mit optimierten Parametern
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=kick_onset_env,
                sr=sr,
                hop_length=self.hop_length,
                backtrack=False,
                pre_max=3,
                post_max=3,
                pre_avg=3,
                post_avg=5,
                delta=0.15,
                wait=4,  # Mindestabstand ~64ms
            )

            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
            return onset_times.tolist()

        except Exception as e:
            logger.warning(f"Kick detection fallback: {e}")
            # Fallback zur alten Methode
            y_kick = self._bandpass_filter(y, sr, 20, 150)
            onset_frames = librosa.onset.onset_detect(y=y_kick, sr=sr, hop_length=self.hop_length)
            return librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length).tolist()

    def _detect_snare_onsets(self, y: np.ndarray, sr: int) -> list[float]:
        """
        Erkennt Snare-Drum Onsets mit kombinierter Body+Transient-Analyse.

        VERBESSERT: Kombiniert Snare-Body (150-400Hz) mit Transient-Snap (1000-4000Hz).

        Args:
            y: Audio-Zeitreihe
            sr: Sample-Rate

        Returns:
            Liste von Snare-Zeiten in Sekunden
        """
        try:
            # Snare-Body (tiefere Frequenzen)
            snare_body_env = librosa.onset.onset_strength(
                y=y,
                sr=sr,
                hop_length=self.hop_length,
                aggregate=np.median,
                fmin=150,
                fmax=400,
                n_mels=32,
            )

            # Transient-Snap (höhere Frequenzen für den "Knack")
            # Percussive component für bessere Transienten
            y_perc = librosa.effects.percussive(y, margin=3)
            transient_env = librosa.onset.onset_strength(
                y=y_perc,
                sr=sr,
                hop_length=self.hop_length,
                aggregate=np.median,
                fmin=1000,
                fmax=min(4000, sr // 2 - 100),  # Nyquist-sicher
                n_mels=32,
            )

            # Kombiniere Body + Snap (60% Body, 40% Transient)
            combined_snare = snare_body_env * 0.6 + transient_env * 0.4

            onset_frames = librosa.onset.onset_detect(
                onset_envelope=combined_snare,
                sr=sr,
                hop_length=self.hop_length,
                backtrack=False,
                pre_max=3,
                post_max=3,
                pre_avg=3,
                post_avg=5,
                delta=0.12,
                wait=6,
            )

            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
            return onset_times.tolist()

        except Exception as e:
            logger.warning(f"Snare detection fallback: {e}")
            y_snare = self._bandpass_filter(y, sr, 150, 2000)
            onset_frames = librosa.onset.onset_detect(y=y_snare, sr=sr, hop_length=self.hop_length)
            return librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length).tolist()

    def _detect_hihat_onsets(self, y: np.ndarray, sr: int) -> list[float]:
        """
        Erkennt Hi-Hat Onsets mit präziser Hochfrequenz-Analyse.

        VERBESSERT: Nutzt Onset-Strength im HiHat-Frequenzbereich (6000-16000Hz).
        Hinweis: Sample-Rate muss mindestens 32000Hz sein für volle Präzision.

        Args:
            y: Audio-Zeitreihe
            sr: Sample-Rate

        Returns:
            Liste von HiHat-Zeiten in Sekunden
        """
        try:
            # Nyquist-sichere Obergrenze
            nyquist = sr // 2
            fmax = min(16000, nyquist - 100)
            fmin = min(6000, fmax - 1000)  # Mindestens 1000Hz Bandbreite

            # BUGFIX #4: Add buffer to prevent Nyquist edge case
            if fmax <= fmin + 1000:
                logger.warning(
                    f"Sample-Rate {sr}Hz zu niedrig für HiHat-Detection "
                    f"(fmin={fmin}, fmax={fmax}), nutze Fallback"
                )
                fmin = 2000
                fmax = nyquist - 100

            hihat_onset_env = librosa.onset.onset_strength(
                y=y,
                sr=sr,
                hop_length=self.hop_length,
                aggregate=np.median,
                fmin=fmin,
                fmax=fmax,
                n_mels=32,
            )

            # Empfindlichere Parameter für schnelle HiHat-Patterns
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=hihat_onset_env,
                sr=sr,
                hop_length=self.hop_length,
                backtrack=False,
                pre_max=2,
                post_max=2,
                pre_avg=2,
                post_avg=3,
                delta=0.08,
                wait=2,  # HiHat kann sehr schnell sein
            )

            # BUGFIX #5: Ensure consistent type (list, not numpy array)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
            return list(onset_times)  # Convert to list explicitly for type consistency

        except Exception as e:
            logger.warning(f"HiHat detection fallback: {e}")
            y_hihat = self._highpass_filter(y, sr, 2000)
            onset_frames = librosa.onset.onset_detect(y=y_hihat, sr=sr, hop_length=self.hop_length)
            return librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length).tolist()

    def _detect_energy_peaks(self, y: np.ndarray, sr: int) -> list[float]:
        """
        Erkennt Energie-Peaks (RMS-basiert).

        Args:
            y: Audio-Zeitreihe
            sr: Sample-Rate

        Returns:
            Liste von Energy-Peak-Zeiten in Sekunden
        """
        # RMS-Energie berechnen
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]

        # Schwellwert: Top X% der Energie-Werte
        threshold = np.percentile(rms, self.energy_percentile)

        # Peaks finden (lokale Maxima über Schwellwert)
        from scipy.signal import find_peaks

        peak_frames, _ = find_peaks(rms, height=threshold, distance=self.hop_length // 2)

        # Zu Zeitpunkten konvertieren
        peak_times = librosa.frames_to_time(peak_frames, sr=sr, hop_length=self.hop_length)

        return peak_times.tolist()

    def _bandpass_filter(
        self, y: np.ndarray, sr: int, low_freq: float, high_freq: float
    ) -> np.ndarray:
        """
        Wendet Band-Pass Filter an.

        Args:
            y: Audio-Zeitreihe
            sr: Sample-Rate
            low_freq: Untere Grenzfrequenz in Hz
            high_freq: Obere Grenzfrequenz in Hz

        Returns:
            Gefiltertes Audio-Signal
        """
        from scipy.signal import butter, filtfilt

        # Nyquist-Frequenz
        nyq = sr / 2.0

        # Normalisierte Frequenzen
        low = low_freq / nyq
        high = high_freq / nyq

        # Butterworth Band-Pass Filter (4. Ordnung)
        b, a = butter(4, [low, high], btype="band")

        # Zero-Phase Filtering (keine Phasenverschiebung)
        y_filtered = filtfilt(b, a, y)

        return y_filtered

    def _highpass_filter(self, y: np.ndarray, sr: int, cutoff_freq: float) -> np.ndarray:
        """
        Wendet High-Pass Filter an.

        Args:
            y: Audio-Zeitreihe
            sr: Sample-Rate
            cutoff_freq: Cutoff-Frequenz in Hz

        Returns:
            Gefiltertes Audio-Signal
        """
        from scipy.signal import butter, filtfilt

        # Nyquist-Frequenz
        nyq = sr / 2.0

        # Normalisierte Cutoff-Frequenz
        cutoff = cutoff_freq / nyq

        # Butterworth High-Pass Filter (4. Ordnung)
        b, a = butter(4, cutoff, btype="high")

        # Zero-Phase Filtering
        y_filtered = filtfilt(b, a, y)

        return y_filtered

    # ========================================================================
    # STEM-BASED ANALYSIS (2025-12-08)
    # Praezisere Trigger-Erkennung fuer DJ-Mixes durch Stem-Separation
    # ========================================================================

    def _init_stem_components(self) -> None:
        """
        Lazy-Loading der Stem-Analyse-Komponenten.

        Initialisiert AudioSeparatorStemExtractor und Analyzer nur bei Bedarf.
        """
        if not STEM_ANALYSIS_AVAILABLE:
            raise RuntimeError(
                "Stem-basierte Analyse nicht verfuegbar. "
                "Installiere: pip install audio-separator[cpu]"
            )

        if self._stem_extractor is None:
            self._stem_extractor = AudioSeparatorStemExtractor(
                cache_dir=self.stem_cache_dir, max_cache_size_gb=50.0
            )

        if self._drums_analyzer is None:
            self._drums_analyzer = DrumsTriggerAnalyzer(sr=self.sr, hop_length=self.hop_length)

        if self._bass_analyzer is None:
            self._bass_analyzer = BassTriggerAnalyzer(sr=self.sr, hop_length=self.hop_length)

        if self._melody_analyzer is None:
            self._melody_analyzer = MelodyTriggerAnalyzer(sr=self.sr, hop_length=self.hop_length)

        if self._stem_merger is None:
            self._stem_merger = StemTriggerMerger(hook_boost_factor=1.5, min_trigger_distance=0.05)

    def analyze_triggers_with_stems(
        self,
        audio_path: str,
        expected_bpm: float | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
        stems_to_use: list[str] | None = None,
    ) -> TriggerAnalysisResult:
        """
        Analysiert Audio mit Stem-Separation fuer praezisere Trigger.

        Trennt Audio in Drums, Bass, Other (Melody/Synth) und analysiert
        jeden Stem direkt ohne Frequenzfilterung. Ergibt bessere
        Kick/Snare/HiHat Erkennung bei DJ-Mixes.

        Fuer EDM: "Other"-Stem = Melodien, Leads, Synthesizer.

        Args:
            audio_path: Pfad zur Audio-Datei
            expected_bpm: Erwartete BPM (optional)
            progress_callback: Optional callback(stage_name, progress_0_to_1)
                              Stages: "stems", "drums", "bass", "melody", "merge"
            stems_to_use: Liste der zu nutzenden Stems ["drums", "bass", "other"]
                          Default: alle drei

        Returns:
            TriggerAnalysisResult mit praeziseren Triggern

        Raises:
            RuntimeError: Wenn Stem-Analyse nicht verfuegbar
            FileNotFoundError: Wenn Audio-Datei nicht existiert
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio-Datei nicht gefunden: {audio_path}")

        if not self.use_stems:
            logger.warning(
                "use_stems=False, aber analyze_triggers_with_stems aufgerufen. Fallback zu Fullmix."
            )
            return self.analyze_triggers(audio_path, expected_bpm)

        # Initialisiere Komponenten
        self._init_stem_components()

        if stems_to_use is None:
            stems_to_use = ["drums", "bass", "other"]  # Other = Melody/Synth fuer EDM

        logger.info(f"Starte Stem-basierte Trigger-Analyse: {audio_path}")
        logger.info(f"Stems: {stems_to_use}")

        # Check Cache fuer stem-basierte Analyse
        cache_key = self._get_cache_key(audio_file) + "_stems"
        cached_result = self._load_trigger_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # 1. Stem-Separation (mit individuellen Progress-Callbacks pro Stem)
        # WICHTIG: Bei AudioTooLongError automatisch auf Fullmix-Analyse fallen
        # Mapping: drums -> drums_sep, bass -> bass_sep, other -> other_sep
        stem_phase_map = {
            "drums": "drums_sep",
            "bass": "bass_sep",
            "other": "other_sep",  # Other = Melody/Synth
        }

        # Initialisiere alle Stem-Progress auf 0
        if progress_callback:
            for stem in stems_to_use:
                phase = stem_phase_map.get(stem, f"{stem}_sep")
                progress_callback(phase, 0.0)

        logger.info("Schritt 1/5: Stem-Separation...")

        def stem_progress_callback(stem_name: str, progress: float):
            """Mappt Stem-Namen auf Progress-Phasen."""
            if progress_callback:
                phase = stem_phase_map.get(stem_name, f"{stem_name}_sep")
                progress_callback(phase, progress)

        # Stem-Separation: SEQUENTIELL - ein Stem nach dem anderen
        # Verhindert ONNX-Crashes bei langen Audio-Dateien durch Memory-Cleanup
        stem_paths = {}
        skipped_stems = []  # CRITICAL FIX: Tracke uebersprungene Stems
        for stem in stems_to_use:
            logger.info(f"Extrahiere Stem: {stem}...")
            try:
                # Nur EINEN Stem pro Aufruf extrahieren
                single_stem_result = self._stem_extractor.separate(
                    audio_file,
                    stems=[stem],  # Nur ein Stem!
                    progress_callback=stem_progress_callback,
                )
                stem_paths.update(single_stem_result)
                logger.info(f"Stem '{stem}' erfolgreich extrahiert")

                # Memory-Cleanup zwischen Stems
                import gc

                gc.collect()

            except AudioTooLongError as e:
                # Graceful Fallback: Bei zu langen Dateien normale Analyse nutzen
                logger.warning(
                    f"Stem-Separation uebersprungen: {e}. "
                    "Fallback auf Fullmix-Analyse (weniger praezise aber sicher)."
                )
                # Alle verbleibenden Stem-Progress als completed markieren
                if progress_callback:
                    for remaining_stem in stems_to_use:
                        phase = stem_phase_map.get(remaining_stem, f"{remaining_stem}_sep")
                        progress_callback(phase, 1.0)
                # Fallback zu normaler Analyse
                return self.analyze_triggers(audio_path, expected_bpm)
            except Exception as e:
                # CRITICAL FIX: Besseres Warning + Tracking statt stilles continue
                logger.warning(
                    f"Stem '{stem}' uebersprungen wegen Fehler: {e}. "
                    f"Analyse wird ohne diesen Stem fortgesetzt."
                )
                skipped_stems.append(stem)
                if progress_callback:
                    phase = stem_phase_map.get(stem, f"{stem}_sep")
                    progress_callback(phase, 1.0)
                continue

        # CRITICAL FIX: Warnung wenn Stems uebersprungen wurden
        if skipped_stems:
            logger.warning(
                f"HINWEIS: {len(skipped_stems)} Stem(s) uebersprungen: {', '.join(skipped_stems)}. "
                f"Die Analyse ist moeglicherweise weniger praezise."
            )

        # Markiere alle Stem-Separationen als abgeschlossen
        if progress_callback:
            for stem in stems_to_use:
                phase = stem_phase_map.get(stem, f"{stem}_sep")
                progress_callback(phase, 1.0)

        # 2. Drums-Analyse
        drums_triggers = None
        if "drums" in stem_paths:
            logger.info("Schritt 2/5: Drums-Analyse...")
            if progress_callback:
                progress_callback("drums", 0.0)

            drums_triggers = self._drums_analyzer.analyze(
                stem_paths["drums"],
                progress_callback=lambda p: (
                    progress_callback("drums", p) if progress_callback else None
                ),
            )

        if progress_callback:
            progress_callback("drums", 1.0)

        # 3. Bass-Analyse
        bass_triggers = None
        if "bass" in stem_paths:
            logger.info("Schritt 3/5: Bass-Analyse...")
            if progress_callback:
                progress_callback("bass", 0.0)

            bass_triggers = self._bass_analyzer.analyze(
                stem_paths["bass"],
                progress_callback=lambda p: (
                    progress_callback("bass", p) if progress_callback else None
                ),
            )

        if progress_callback:
            progress_callback("bass", 1.0)

        # 4. Melody/Synth-Analyse (Other-Stem)
        melody_triggers = None
        if "other" in stem_paths:
            logger.info("Schritt 4/5: Melody/Synth-Analyse...")
            if progress_callback:
                progress_callback("melody", 0.0)

            melody_triggers = self._melody_analyzer.analyze(
                stem_paths["other"],
                progress_callback=lambda p: (
                    progress_callback("melody", p) if progress_callback else None
                ),
            )

        if progress_callback:
            progress_callback("melody", 1.0)

        # 5. Merge Triggers
        logger.info("Schritt 5/5: Trigger zusammenfuehren...")
        if progress_callback:
            progress_callback("merge", 0.0)

        merged = self._stem_merger.merge(
            drums=drums_triggers, bass=bass_triggers, melody=melody_triggers
        )

        # Audio-Dauer und BPM aus Fullmix holen
        y, sr = librosa.load(audio_path, sr=self.sr, duration=30)  # Nur 30s fuer BPM
        full_duration = librosa.get_duration(path=audio_path)

        if expected_bpm:
            bpm = expected_bpm
        else:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
            bpm = float(tempo)

        # Beat-Times aus Fullmix (BPM-Grid bleibt gleich)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        _, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=self.hop_length,
            start_bpm=bpm,
            tightness=100,
        )
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length).tolist()

        # Extrapoliere Beats fuer gesamte Dauer
        # FIX #4: Erweiterte BPM-Validierung (Division-by-Zero, NaN, Infinity, Endlosschleife)
        if beat_times and full_duration > beat_times[-1] and bpm > 0 and np.isfinite(bpm):
            beat_interval = 60.0 / bpm
            # Zusätzliche Validierung: beat_interval muss positiv und endlich sein
            if (
                beat_interval > 0 and np.isfinite(beat_interval) and beat_interval >= 0.01
            ):  # Min 10ms zwischen Beats
                last_beat = beat_times[-1]
                # Max-Iterations-Check: Verhindert Endlosschleife bei extremen Werten
                max_iterations = min(10000, int((full_duration - last_beat) / beat_interval) + 1)
                iterations = 0
                while last_beat + beat_interval < full_duration and iterations < max_iterations:
                    last_beat += beat_interval
                    beat_times.append(last_beat)
                    iterations += 1
                if iterations >= max_iterations:
                    logger.warning(f"Beat-Extrapolation abgebrochen nach {iterations} Iterationen")
            else:
                logger.warning(
                    f"Beat-Extrapolation uebersprungen: ungültiges beat_interval={beat_interval}"
                )

        if progress_callback:
            progress_callback("merge", 0.5)

        # Konvertiere zu TriggerAnalysisResult
        kick_times = merged.get("kick", (np.array([]), np.array([])))[0].tolist()
        snare_times = merged.get("snare", (np.array([]), np.array([])))[0].tolist()
        hihat_times = merged.get("hihat", (np.array([]), np.array([])))[0].tolist()

        # Synth-Stabs und Melody-Hooks (aus Other-Stem)
        synth_stabs = merged.get("synth_stab", (np.array([]), np.array([])))[0].tolist()
        melody_hooks = merged.get("melody_hook", (np.array([]), np.array([])))[0].tolist()

        # Onset-Times = alle Drum-Onsets + Synth-Stabs kombiniert
        all_onsets = sorted(set(kick_times + snare_times + hihat_times + synth_stabs))

        # Energy-Times = Bass-Drops + Melody-Hooks (beides sind energetische Momente)
        bass_drops = merged.get("bass_drop", (np.array([]), np.array([])))[0].tolist()
        energy_times = sorted(set(bass_drops + melody_hooks))

        if progress_callback:
            progress_callback("merge", 1.0)

        logger.info(
            f"Stem-Trigger-Analyse abgeschlossen: {len(beat_times)} beats, "
            f"{len(kick_times)} kicks (stem), {len(snare_times)} snares (stem), "
            f"{len(hihat_times)} hihats (stem), {len(synth_stabs)} synth-stabs, "
            f"{len(melody_hooks)} melody-hooks, {len(bass_drops)} bass-drops"
        )

        result = TriggerAnalysisResult(
            beat_times=beat_times,
            onset_times=all_onsets,
            kick_times=kick_times,
            snare_times=snare_times,
            hihat_times=hihat_times,
            energy_times=energy_times,
            bpm=bpm,
            duration=full_duration,
            sample_rate=sr,
        )

        # Cache speichern
        self._save_trigger_cache(cache_key, result)

        return result

    def is_stem_analysis_available(self) -> bool:
        """Prueft ob Stem-basierte Analyse verfuegbar ist."""
        return STEM_ANALYSIS_AVAILABLE

    def get_stem_cache_size(self) -> int:
        """Gibt Groesse des Stem-Caches in Bytes zurueck."""
        if self._stem_extractor is None:
            return 0
        return self._stem_extractor.get_cache_size()

    def clear_stem_cache(self) -> None:
        """Loescht den Stem-Cache."""
        if self._stem_extractor is not None:
            self._stem_extractor.clear_cache()
        elif self.stem_cache_dir.exists():
            import shutil

            shutil.rmtree(self.stem_cache_dir)
            self.stem_cache_dir.mkdir(parents=True, exist_ok=True)
