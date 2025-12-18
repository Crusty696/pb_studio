"""
Clip Matcher - Intelligente Clip-Auswahl basierend auf Analyse-Daten.

Verbindet die Video-Analyse mit der Pacing Engine fuer:
- Mood-basierte Clip-Auswahl (MoodMatcher)
- Motion-basierte Clip-Auswahl (MotionMatcher)
- Energie-Kurven Matching (EnergyMatcher)
- Kombinierte intelligente Auswahl (SmartMatcher)

Author: PB_studio Development Team
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..database.connection import managed_session
from ..database.models import VideoClip
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MatchStrategy(Enum):
    """Strategie fuer Clip-Matching."""

    EXACT = "exact"  # Exakte Uebereinstimmung
    SIMILAR = "similar"  # Aehnliche Werte
    CONTRAST = "contrast"  # Gegensaetzliche Werte (fuer Abwechslung)


@dataclass
class ClipScore:
    """Score fuer einen Clip-Kandidaten."""

    clip_id: int
    clip_name: str
    file_path: str
    duration: float
    score: float  # 0-1, hoeher = bessere Uebereinstimmung
    match_reasons: list[str]  # Gruende fuer den Score

    def __repr__(self):
        return f"ClipScore(id={self.clip_id}, score={self.score:.2f}, reasons={len(self.match_reasons)})"


class ClipAnalysisLoader:
    """Laedt Clip-Analyse-Daten aus der Datenbank."""

    def __init__(self):
        self._cache: dict[int, dict[str, Any]] = {}

    def get_clip_analysis(self, clip_id: int) -> dict[str, Any] | None:
        """
        Laedt Analyse-Daten fuer einen Clip.

        Args:
            clip_id: Clip ID

        Returns:
            Dict mit Analyse-Daten oder None
        """
        if clip_id in self._cache:
            return self._cache[clip_id]

        # CRITICAL-06 FIX: Use managed_session context manager to prevent session leaks
        try:
            with managed_session() as db:
                clip = db.query(VideoClip).filter(VideoClip.id == clip_id).first()

                if not clip:
                    return None

                analysis = {
                    "id": clip.id,
                    "name": clip.name,
                    "file_path": clip.file_path,
                    "duration": clip.duration_seconds or 0.0,
                    "is_analyzed": clip.is_analyzed or False,
                }

                # Parse JSON Felder
                for field in [
                    "color_analysis",
                    "motion_analysis",
                    "scene_analysis",
                    "mood_analysis",
                    "style_analysis",
                    "object_analysis",
                ]:
                    value = getattr(clip, field, None)
                    if value:
                        try:
                            analysis[field.replace("_analysis", "")] = (
                                json.loads(value) if isinstance(value, str) else value
                            )
                        except (json.JSONDecodeError, TypeError):
                            pass

                self._cache[clip_id] = analysis
                return analysis

        except Exception as e:
            logger.error(f"Failed to load clip analysis {clip_id}: {e}")
            return None

    def get_all_analyzed_clips(self, limit: int = 1000) -> list[dict[str, Any]]:
        """
        Laedt alle analysierten Clips.

        Args:
            limit: Maximum number of clips to load (default 1000)

        Returns:
            Liste von Clip-Analyse-Dicts
        """
        # CRITICAL-07 FIX: Use managed_session + pagination to prevent session leaks
        try:
            with managed_session() as db:
                clips = (
                    db.query(VideoClip)
                    .filter(VideoClip.is_analyzed == True, VideoClip.is_available == True)
                    .limit(limit)
                    .all()
                )

                results = []
                for clip in clips:
                    # Use cache if available, otherwise load from db
                    analysis = self.get_clip_analysis(clip.id)
                    if analysis:
                        results.append(analysis)

                return results

        except Exception as e:
            logger.error(f"Failed to load analyzed clips: {e}")
            return []

    def clear_cache(self):
        """Leert den Cache."""
        self._cache.clear()


class MoodMatcher:
    """
    Findet Clips basierend auf Mood/Stimmung.

    Nutzt mood_analysis Daten:
    - moods: Liste von Stimmungen (ENERGETIC, CALM, etc.)
    - energy: Energie-Wert (0-1)
    - warm_ratio / cool_ratio: Farbtemperatur
    """

    MOOD_COMPATIBILITY = {
        # Mood -> kompatible Moods (sortiert nach Kompatibilitaet)
        "ENERGETIC": ["EUPHORIC", "AGGRESSIVE", "CHEERFUL", "TENSE"],
        "CALM": ["PEACEFUL", "DREAMY", "MELANCHOLIC"],
        "DARK": ["MYSTERIOUS", "TENSE", "MELANCHOLIC", "AGGRESSIVE"],
        "BRIGHT": ["CHEERFUL", "EUPHORIC", "PEACEFUL"],
        "MELANCHOLIC": ["CALM", "DARK", "DREAMY", "MYSTERIOUS"],
        "EUPHORIC": ["ENERGETIC", "CHEERFUL", "BRIGHT"],
        "AGGRESSIVE": ["ENERGETIC", "TENSE", "DARK"],
        "PEACEFUL": ["CALM", "DREAMY", "BRIGHT"],
        "MYSTERIOUS": ["DARK", "MELANCHOLIC", "TENSE"],
        "CHEERFUL": ["BRIGHT", "EUPHORIC", "ENERGETIC"],
        "TENSE": ["DARK", "AGGRESSIVE", "MYSTERIOUS", "ENERGETIC"],
        "DREAMY": ["CALM", "PEACEFUL", "MELANCHOLIC"],
        "COOL": ["CALM", "MYSTERIOUS", "MELANCHOLIC"],
        "WARM": ["CHEERFUL", "ENERGETIC", "BRIGHT"],
    }

    def __init__(self, loader: ClipAnalysisLoader | None = None):
        self.loader = loader or ClipAnalysisLoader()

    def find_matching_clips(
        self,
        target_mood: str,
        target_energy: float | None = None,
        strategy: MatchStrategy = MatchStrategy.SIMILAR,
        limit: int = 10,
        exclude_ids: list[int] | None = None,
        candidate_ids: list[int] | None = None,
    ) -> list[ClipScore]:
        """
        Findet Clips die zur Ziel-Stimmung passen.

        Args:
            target_mood: Ziel-Stimmung (ENERGETIC, CALM, etc.)
            target_energy: Optionaler Ziel-Energie-Wert (0-1)
            strategy: Match-Strategie
            limit: Maximale Anzahl Ergebnisse
            exclude_ids: Clip-IDs die ausgeschlossen werden sollen
            candidate_ids: Optional: Nur diese IDs beruecksichtigen

        Returns:
            Liste von ClipScore sortiert nach Score (hoechster zuerst)
        """
        exclude_ids = exclude_ids or []

        # Determine source clips
        if candidate_ids is not None:
            clips = []
            for cid in candidate_ids:
                c_data = self.loader.get_clip_analysis(cid)
                if c_data:
                    clips.append(c_data)
        else:
            clips = self.loader.get_all_analyzed_clips()

        scores = []

        for clip in clips:
            if clip["id"] in exclude_ids:
                continue

            mood_data = clip.get("mood", {})
            if not mood_data:
                continue

            score, reasons = self._calculate_mood_score(
                mood_data, target_mood, target_energy, strategy
            )

            if score > 0:
                scores.append(
                    ClipScore(
                        clip_id=clip["id"],
                        clip_name=clip["name"],
                        file_path=clip["file_path"],
                        duration=clip["duration"],
                        score=score,
                        match_reasons=reasons,
                    )
                )

        # Sortieren nach Score (absteigend)
        scores.sort(key=lambda x: x.score, reverse=True)

        return scores[:limit]

    def _calculate_mood_score(
        self,
        mood_data: dict,
        target_mood: str,
        target_energy: float | None,
        strategy: MatchStrategy,
    ) -> tuple[float, list[str]]:
        """Berechnet Score fuer Mood-Match."""
        score = 0.0
        reasons = []

        clip_moods = mood_data.get("moods", [])
        clip_energy = mood_data.get("energy", 0.5)

        # Exakte Mood-Uebereinstimmung
        if target_mood in clip_moods:
            score += 0.5
            reasons.append(f"Exact mood match: {target_mood}")

        # Kompatible Moods
        elif target_mood in self.MOOD_COMPATIBILITY:
            compatible = self.MOOD_COMPATIBILITY[target_mood]
            for mood in clip_moods:
                if mood in compatible:
                    score += 0.3
                    reasons.append(f"Compatible mood: {mood}")
                    break

        # Energie-Match
        if target_energy is not None:
            if strategy == MatchStrategy.EXACT:
                energy_diff = abs(clip_energy - target_energy)
                if energy_diff < 0.1:
                    score += 0.3
                    reasons.append(f"Exact energy match: {clip_energy:.2f}")
            elif strategy == MatchStrategy.SIMILAR:
                energy_diff = abs(clip_energy - target_energy)
                energy_score = max(0, 0.3 - energy_diff * 0.5)
                score += energy_score
                if energy_score > 0.1:
                    reasons.append(f"Similar energy: {clip_energy:.2f}")
            elif strategy == MatchStrategy.CONTRAST:
                # Gegensaetzliche Energie
                energy_diff = abs(clip_energy - target_energy)
                if energy_diff > 0.5:
                    score += 0.3
                    reasons.append(f"Contrasting energy: {clip_energy:.2f}")

        # Bonus fuer mehrere passende Moods
        if len([m for m in clip_moods if m in self.MOOD_COMPATIBILITY.get(target_mood, [])]) > 1:
            score += 0.2
            reasons.append("Multiple mood matches")

        return min(score, 1.0), reasons


class MotionMatcher:
    """
    Findet Clips basierend auf Bewegungscharakteristiken.

    Nutzt motion_analysis Daten:
    - motion_type: STATIC, SLOW, MEDIUM, FAST, EXTREME
    - motion_score: 0-1
    - camera_motion: STATIC_CAM, PAN_LEFT, etc.
    """

    MOTION_HIERARCHY = ["STATIC", "SLOW", "MEDIUM", "FAST", "EXTREME"]

    def __init__(self, loader: ClipAnalysisLoader | None = None):
        self.loader = loader or ClipAnalysisLoader()

    def find_matching_clips(
        self,
        target_motion: str,
        target_score: float | None = None,
        camera_motion: str | None = None,
        strategy: MatchStrategy = MatchStrategy.SIMILAR,
        limit: int = 10,
        exclude_ids: list[int] | None = None,
        candidate_ids: list[int] | None = None,
    ) -> list[ClipScore]:
        """
        Findet Clips die zur Ziel-Bewegung passen.

        Args:
            target_motion: Ziel-Bewegungstyp (STATIC, SLOW, MEDIUM, FAST, EXTREME)
            target_score: Optionaler Ziel-Motion-Score (0-1)
            camera_motion: Optionaler Kamera-Bewegungstyp
            strategy: Match-Strategie
            limit: Maximale Anzahl Ergebnisse
            exclude_ids: Clip-IDs die ausgeschlossen werden sollen
            candidate_ids: Optional: Nur diese IDs beruecksichtigen

        Returns:
            Liste von ClipScore sortiert nach Score
        """
        exclude_ids = exclude_ids or []

        # Determine source clips
        if candidate_ids is not None:
            clips = []
            for cid in candidate_ids:
                c_data = self.loader.get_clip_analysis(cid)
                if c_data:
                    clips.append(c_data)
        else:
            clips = self.loader.get_all_analyzed_clips()

        scores = []

        for clip in clips:
            if clip["id"] in exclude_ids:
                continue

            motion_data = clip.get("motion", {})
            if not motion_data:
                continue

            score, reasons = self._calculate_motion_score(
                motion_data, target_motion, target_score, camera_motion, strategy
            )

            if score > 0:
                scores.append(
                    ClipScore(
                        clip_id=clip["id"],
                        clip_name=clip["name"],
                        file_path=clip["file_path"],
                        duration=clip["duration"],
                        score=score,
                        match_reasons=reasons,
                    )
                )

        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[:limit]

    def _calculate_motion_score(
        self,
        motion_data: dict,
        target_motion: str,
        target_score: float | None,
        camera_motion: str | None,
        strategy: MatchStrategy,
    ) -> tuple[float, list[str]]:
        """Berechnet Score fuer Motion-Match."""
        score = 0.0
        reasons = []

        clip_motion = motion_data.get("motion_type", "")
        clip_score = motion_data.get("motion_score", 0.0)
        clip_camera = motion_data.get("camera_motion", "")

        # Motion Type Match
        if target_motion in self.MOTION_HIERARCHY and clip_motion in self.MOTION_HIERARCHY:
            target_idx = self.MOTION_HIERARCHY.index(target_motion)
            clip_idx = self.MOTION_HIERARCHY.index(clip_motion)

            if strategy == MatchStrategy.EXACT:
                if clip_motion == target_motion:
                    score += 0.5
                    reasons.append(f"Exact motion: {clip_motion}")
            elif strategy == MatchStrategy.SIMILAR:
                distance = abs(target_idx - clip_idx)
                if distance == 0:
                    score += 0.5
                    reasons.append(f"Exact motion: {clip_motion}")
                elif distance == 1:
                    score += 0.35
                    reasons.append(f"Similar motion: {clip_motion}")
                elif distance == 2:
                    score += 0.2
                    reasons.append(f"Related motion: {clip_motion}")
            elif strategy == MatchStrategy.CONTRAST:
                distance = abs(target_idx - clip_idx)
                if distance >= 2:
                    score += 0.4
                    reasons.append(f"Contrasting motion: {clip_motion}")

        # Motion Score Match
        if target_score is not None:
            score_diff = abs(clip_score - target_score)
            if strategy in [MatchStrategy.EXACT, MatchStrategy.SIMILAR]:
                score_bonus = max(0, 0.3 - score_diff * 0.5)
                score += score_bonus
                if score_bonus > 0.1:
                    reasons.append(f"Motion score: {clip_score:.2f}")

        # Camera Motion Match
        if camera_motion and clip_camera:
            if clip_camera == camera_motion:
                score += 0.2
                reasons.append(f"Camera motion: {clip_camera}")

        return min(score, 1.0), reasons


class EnergyMatcher:
    """
    Matching basierend auf Audio-Energie-Kurven.

    Mapped Audio-Energie auf Video-Clips:
    - Hohe Energie -> schnelle/bewegte Clips
    - Niedrige Energie -> ruhige/statische Clips
    """

    def __init__(self, loader: ClipAnalysisLoader | None = None):
        self.loader = loader or ClipAnalysisLoader()

    def match_energy_curve(
        self, energy_values: list[float], timestamps: list[float], limit_per_segment: int = 3
    ) -> list[tuple[float, list[ClipScore]]]:
        """
        Matched Clips zu einer Energie-Kurve.

        Args:
            energy_values: Liste von Energie-Werten (0-1)
            timestamps: Korrespondierende Zeitstempel
            limit_per_segment: Max Clips pro Energie-Segment

        Returns:
            Liste von (timestamp, ClipScore[]) Tupeln
        """
        results = []

        for i, (energy, timestamp) in enumerate(zip(energy_values, timestamps)):
            # Bestimme Ziel-Motion basierend auf Energie
            target_motion = self._energy_to_motion(energy)

            # Verwende Mood-Matching auch basierend auf Energie
            target_mood = self._energy_to_mood(energy)

            # Kombiniere Motion und Mood Matching
            motion_matcher = MotionMatcher(self.loader)
            mood_matcher = MoodMatcher(self.loader)

            motion_clips = motion_matcher.find_matching_clips(
                target_motion=target_motion,
                target_score=energy,
                strategy=MatchStrategy.SIMILAR,
                limit=limit_per_segment,
            )

            mood_clips = mood_matcher.find_matching_clips(
                target_mood=target_mood,
                target_energy=energy,
                strategy=MatchStrategy.SIMILAR,
                limit=limit_per_segment,
            )

            # Kombiniere Scores
            combined = self._combine_scores(motion_clips, mood_clips)
            results.append((timestamp, combined[:limit_per_segment]))

        return results

    def _energy_to_motion(self, energy: float) -> str:
        """Konvertiert Energie zu Motion-Type."""
        if energy < 0.2:
            return "STATIC"
        elif energy < 0.4:
            return "SLOW"
        elif energy < 0.6:
            return "MEDIUM"
        elif energy < 0.8:
            return "FAST"
        else:
            return "EXTREME"

    def _energy_to_mood(self, energy: float) -> str:
        """Konvertiert Energie zu Mood."""
        if energy < 0.3:
            return "CALM"
        elif energy < 0.6:
            return "PEACEFUL" if energy < 0.45 else "CHEERFUL"
        else:
            return "ENERGETIC"

    def _combine_scores(
        self, motion_scores: list[ClipScore], mood_scores: list[ClipScore]
    ) -> list[ClipScore]:
        """Kombiniert Motion und Mood Scores."""
        combined: dict[int, ClipScore] = {}

        for score in motion_scores:
            combined[score.clip_id] = ClipScore(
                clip_id=score.clip_id,
                clip_name=score.clip_name,
                file_path=score.file_path,
                duration=score.duration,
                score=score.score * 0.6,  # 60% Gewicht fuer Motion
                match_reasons=score.match_reasons.copy(),
            )

        for score in mood_scores:
            if score.clip_id in combined:
                existing = combined[score.clip_id]
                existing.score += score.score * 0.4  # 40% Gewicht fuer Mood
                existing.match_reasons.extend(score.match_reasons)
            else:
                combined[score.clip_id] = ClipScore(
                    clip_id=score.clip_id,
                    clip_name=score.clip_name,
                    file_path=score.file_path,
                    duration=score.duration,
                    score=score.score * 0.4,
                    match_reasons=score.match_reasons.copy(),
                )

        # Sortieren und zurueckgeben
        result = list(combined.values())
        result.sort(key=lambda x: x.score, reverse=True)
        return result


class SmartMatcher:
    """
    Kombiniertes intelligentes Matching mit mehreren Kriterien.

    Unterstuetzt:
    - Multi-Kriterien Matching
    - Gewichtung verschiedener Faktoren
    - Vermeidung von Wiederholungen
    """

    def __init__(self, loader: ClipAnalysisLoader | None = None):
        self.loader = loader or ClipAnalysisLoader()
        self.mood_matcher = MoodMatcher(self.loader)
        self.motion_matcher = MotionMatcher(self.loader)
        self._used_clips: list[int] = []

    def find_best_clip(
        self,
        target_mood: str | None = None,
        target_motion: str | None = None,
        target_energy: float | None = None,
        min_duration: float = 0.5,
        max_duration: float | None = None,
        avoid_recent: int = 5,  # Vermeide die letzten N verwendeten Clips
        weights: dict[str, float] | None = None,
        candidate_ids: list[int] | None = None,
    ) -> ClipScore | None:
        """
        Findet den besten Clip basierend auf mehreren Kriterien.

        Args:
            target_mood: Ziel-Stimmung
            target_motion: Ziel-Bewegung
            target_energy: Ziel-Energie (0-1)
            min_duration: Minimale Clip-Dauer
            max_duration: Maximale Clip-Dauer
            avoid_recent: Vermeide N zuletzt verwendete Clips
            weights: Gewichtung {'mood': 0.4, 'motion': 0.6}
            candidate_ids: Optional: Nur aus diesen IDs waehlen

        Returns:
            Bester ClipScore oder None
        """
        weights = weights or {"mood": 0.5, "motion": 0.5}

        # Zu vermeidende Clip-IDs
        exclude_ids = self._used_clips[-avoid_recent:] if avoid_recent > 0 else []

        candidates: dict[int, ClipScore] = {}

        # Mood-basierte Kandidaten
        if target_mood:
            mood_clips = self.mood_matcher.find_matching_clips(
                target_mood=target_mood,
                target_energy=target_energy,
                limit=20,
                exclude_ids=exclude_ids,
                candidate_ids=candidate_ids,
            )
            for clip in mood_clips:
                if self._check_duration(clip.duration, min_duration, max_duration):
                    candidates[clip.clip_id] = ClipScore(
                        clip_id=clip.clip_id,
                        clip_name=clip.clip_name,
                        file_path=clip.file_path,
                        duration=clip.duration,
                        score=clip.score * weights.get("mood", 0.5),
                        match_reasons=clip.match_reasons.copy(),
                    )

        # Motion-basierte Kandidaten
        if target_motion:
            motion_clips = self.motion_matcher.find_matching_clips(
                target_motion=target_motion,
                target_score=target_energy,
                limit=20,
                exclude_ids=exclude_ids,
                candidate_ids=candidate_ids,
            )
            for clip in motion_clips:
                if not self._check_duration(clip.duration, min_duration, max_duration):
                    continue

                if clip.clip_id in candidates:
                    existing = candidates[clip.clip_id]
                    existing.score += clip.score * weights.get("motion", 0.5)
                    existing.match_reasons.extend(clip.match_reasons)
                else:
                    candidates[clip.clip_id] = ClipScore(
                        clip_id=clip.clip_id,
                        clip_name=clip.clip_name,
                        file_path=clip.file_path,
                        duration=clip.duration,
                        score=clip.score * weights.get("motion", 0.5),
                        match_reasons=clip.match_reasons.copy(),
                    )

        # Fallback: Wenn keine Mood/Motion gefunden, aber Candidates existieren
        # Versuche einfach einen validen Candidate zu finden (z.B. nach Duration)
        if not candidates and candidate_ids:
            # Logic to just pick a valid clip if 'matching' failed but we need ONE?
            # For now, return None and let caller handle fallback.
            pass

        if not candidates:
            return None

        # Besten Clip auswaehlen
        best = max(candidates.values(), key=lambda x: x.score)

        # Als verwendet markieren
        self._used_clips.append(best.clip_id)

        return best

    def _check_duration(self, duration: float, min_dur: float, max_dur: float | None) -> bool:
        """Prueft ob Dauer in erlaubtem Bereich."""
        if duration < min_dur:
            return False
        if max_dur and duration > max_dur:
            return False
        return True

    def reset_used_clips(self):
        """Setzt die Liste verwendeter Clips zurueck."""
        self._used_clips.clear()

    def get_used_clips(self) -> list[int]:
        """Gibt verwendete Clips zurueck."""
        return self._used_clips.copy()
