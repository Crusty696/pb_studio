"""
Rekordbox XML Importer für PB_studio

Importiert:
- Track-Metadaten (BPM, Key, Titel, Artist)
- Beatgrid-Informationen
- Cue-Points und Memory-Cues
- Playlists und Folders
"""

# SECURITY: Use defusedxml to prevent XXE (External Entity Injection) attacks
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Standard library for type hints only (defusedxml doesn't export Element type)
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as DefusedET

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RekordboxTrack:
    """Repräsentiert einen Track aus Rekordbox."""

    track_id: str
    name: str
    artist: str
    file_path: str
    bpm: float
    key: str | None = None
    duration_ms: int = 0
    beatgrid_offset: float = 0.0
    cue_points: list[dict[str, Any]] = field(default_factory=list)


class RekordboxImporter:
    """Importiert Rekordbox XML-Bibliotheken."""

    def __init__(self, xml_path: str):
        self.xml_path = Path(xml_path)
        self.tracks: list[RekordboxTrack] = []
        self.playlists: dict[str, list[str]] = {}

    def parse(self) -> bool:
        """
        Parst die XML-Datei.

        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # SECURITY: Use defusedxml.parse() for safe XML parsing
            tree = DefusedET.parse(self.xml_path)
            root = tree.getroot()

            # COLLECTION parsen
            collection = root.find(".//COLLECTION")
            if collection is not None:
                self._parse_collection(collection)

            # PLAYLISTS parsen
            playlists = root.find(".//PLAYLISTS")
            if playlists is not None:
                self._parse_playlists(playlists)

            logger.info(
                f"Rekordbox Import: {len(self.tracks)} Tracks, {len(self.playlists)} Playlists"
            )
            return True

        except Exception as e:
            logger.error(f"Rekordbox XML Parse-Fehler: {e}")
            return False

    def _parse_collection(self, collection: Element):
        """Parst die Track-Collection."""
        for track_elem in collection.findall("TRACK"):
            track = self._parse_track(track_elem)
            if track:
                self.tracks.append(track)

    def _parse_track(self, elem: Element) -> RekordboxTrack | None:
        """
        Parst einen einzelnen Track.

        Args:
            elem: Track XML Element

        Returns:
            RekordboxTrack oder None bei Fehler
        """
        try:
            # Basis-Attribute
            track_id = elem.get("TrackID", "")
            name = elem.get("Name", "Unknown")
            artist = elem.get("Artist", "Unknown")
            location = elem.get("Location", "")

            # BPM (als String mit Dezimalstellen)
            bpm_str = elem.get("AverageBpm", "120.00")
            bpm = float(bpm_str)

            # Key
            key = elem.get("Tonality")

            # Duration in ms
            duration_str = elem.get("TotalTime", "0")
            duration_ms = int(float(duration_str) * 1000)

            # File-Pfad dekodieren (URL-encoded)
            from urllib.parse import unquote

            file_path = unquote(location.replace("file://localhost/", ""))

            # Cue-Points
            cue_points = []
            for pos_mark in elem.findall("POSITION_MARK"):
                cue = {
                    "name": pos_mark.get("Name", ""),
                    "type": pos_mark.get("Type", "0"),
                    "start": float(pos_mark.get("Start", "0")),
                    "num": int(pos_mark.get("Num", "-1")),
                }
                cue_points.append(cue)

            # TEMPO für Beatgrid-Offset
            tempo_elem = elem.find("TEMPO")
            beatgrid_offset = 0.0
            if tempo_elem is not None:
                beatgrid_offset = float(tempo_elem.get("Inizio", "0"))

            return RekordboxTrack(
                track_id=track_id,
                name=name,
                artist=artist,
                file_path=file_path,
                bpm=bpm,
                key=key,
                duration_ms=duration_ms,
                beatgrid_offset=beatgrid_offset,
                cue_points=cue_points,
            )

        except Exception as e:
            logger.warning(f"Track-Parse-Fehler: {e}")
            return None

    def _parse_playlists(self, playlists: Element):
        """Parst Playlists."""
        for node in playlists.findall(".//NODE[@Type='1']"):
            name = node.get("Name", "Unknown Playlist")
            track_ids = []

            for track in node.findall("TRACK"):
                track_id = track.get("Key")
                if track_id:
                    track_ids.append(track_id)

            if track_ids:
                self.playlists[name] = track_ids

    def get_track_by_id(self, track_id: str) -> RekordboxTrack | None:
        """
        Findet Track nach ID.

        Args:
            track_id: Rekordbox Track ID

        Returns:
            RekordboxTrack oder None
        """
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None

    def import_to_database(self, project_id: int) -> int:
        """
        Importiert Tracks in die Projekt-Datenbank.

        Args:
            project_id: ID des Projekts

        Returns:
            Anzahl importierter Tracks
        """
        from ..database.crud import create_audio_track, update_audio_track

        imported = 0
        for track in self.tracks:
            try:
                # Audio-Track erstellen
                audio_track = create_audio_track(
                    project_id=project_id,
                    name=f"{track.artist} - {track.name}",
                    file_path=track.file_path,
                )

                if audio_track:
                    # BPM und Metadaten setzen
                    update_audio_track(
                        audio_track.id,
                        bpm=track.bpm,
                        duration=track.duration_ms / 1000.0,
                        is_analyzed=True,  # Rekordbox hat bereits analysiert
                    )
                    imported += 1
                    logger.debug(f"Importiert: {track.name} (BPM: {track.bpm})")

            except Exception as e:
                logger.warning(f"Import-Fehler für Track '{track.name}': {e}")

        logger.info(
            f"Rekordbox Import: {imported}/{len(self.tracks)} Tracks erfolgreich importiert"
        )
        return imported
