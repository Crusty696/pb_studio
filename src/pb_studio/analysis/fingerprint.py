"""
Content Fingerprint System fuer Clip-Wiedererkennung.

Erstellt einen eindeutigen Fingerprint basierend auf:
- Video-Metadaten (FPS, Frame-Count, Aufloesung)
- Sample-Frames aus dem Video

Der Fingerprint ueberlebt:
- Umbenennung
- Verschiebung in anderen Ordner
- Kopie auf anderes Laufwerk
"""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from ..utils.logger import get_logger

logger = get_logger()


class ContentFingerprint:
    """Berechnet und verwaltet Content-Fingerprints fuer Videos."""

    def __init__(self):
        self.sample_size = (8, 8)  # Kleine Sample-Groesse fuer schnelle Berechnung
        self.num_samples = 4  # Anzahl Frame-Samples

    def compute(self, video_path: str) -> str | None:
        """
        Berechnet Content-Fingerprint fuer ein Video.

        Args:
            video_path: Pfad zum Video

        Returns:
            16-Zeichen Hex-String oder None bei Fehler
        """
        # PERF-02 FIX: Use context manager to ensure VideoCapture is released
        from ..utils.video_utils import open_video

        try:
            path = Path(video_path)
            if not path.exists():
                logger.warning(f"Video nicht gefunden: {video_path}")
                return None

            with open_video(str(path)) as cap:
                if not cap.isOpened():
                    logger.warning(f"Konnte Video nicht oeffnen: {video_path}")
                    return None

                # Metadaten sammeln
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if frame_count <= 0:
                    logger.warning(f"Keine Frames in Video: {video_path}")
                    return None

                # Sample-Positionen berechnen
                positions = self._get_sample_positions(frame_count)

                # Frames samplen
                samples = []
                for pos in positions:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Auf kleine Groesse skalieren
                        small = cv2.resize(frame, self.sample_size, interpolation=cv2.INTER_AREA)
                        samples.append(small.tobytes())

            # VideoCapture automatically released by context manager

            if len(samples) < 2:
                logger.warning(f"Zu wenig Frames gesampelt: {video_path}")
                return None

            # Fingerprint aus Metadaten + Samples erstellen
            data = f"{fps:.2f}_{frame_count}_{width}x{height}_".encode()
            data += b"".join(samples)

            # SHA256 Hash, gekuerzt auf 16 Zeichen
            fingerprint = hashlib.sha256(data).hexdigest()[:16]

            logger.debug(f"Fingerprint berechnet: {fingerprint} fuer {path.name}")
            return fingerprint

        except Exception as e:
            logger.error(f"Fehler bei Fingerprint-Berechnung fuer {video_path}: {e}")
            return None

    def _get_sample_positions(self, frame_count: int) -> list:
        """Berechnet Frame-Positionen fuer Sampling."""
        if frame_count <= self.num_samples:
            return list(range(frame_count))

        # Gleichmaessig verteilt: 0%, 25%, 50%, 75%
        return [0, frame_count // 4, frame_count // 2, 3 * frame_count // 4]

    def compute_batch(self, video_paths: list, max_workers: int = 4) -> dict:
        """
        Berechnet Fingerprints fuer mehrere Videos parallel.

        Args:
            video_paths: Liste von Video-Pfaden
            max_workers: Maximale Anzahl paralleler Workers

        Returns:
            Dict {pfad: fingerprint}
        """
        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.compute, path): path for path in video_paths}

            for future in futures:
                path = futures[future]
                try:
                    fingerprint = future.result()
                    results[path] = fingerprint
                except Exception as e:
                    logger.error(f"Batch-Fingerprint Fehler fuer {path}: {e}")
                    results[path] = None

        return results

    def verify(self, video_path: str, expected_fingerprint: str) -> bool:
        """
        Verifiziert ob ein Video den erwarteten Fingerprint hat.

        Args:
            video_path: Pfad zum Video
            expected_fingerprint: Erwarteter Fingerprint

        Returns:
            True wenn Fingerprints uebereinstimmen
        """
        actual = self.compute(video_path)
        if actual is None:
            return False
        return actual == expected_fingerprint

    def find_by_fingerprint(self, fingerprint: str, search_paths: list) -> str | None:
        """
        Sucht ein Video mit gegebenem Fingerprint in mehreren Pfaden.

        Args:
            fingerprint: Gesuchter Fingerprint
            search_paths: Liste von Ordnern zum Durchsuchen

        Returns:
            Gefundener Pfad oder None
        """
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"}

        for search_path in search_paths:
            path = Path(search_path)
            if not path.exists():
                continue

            # Alle Videos im Ordner durchsuchen
            for video_file in path.rglob("*"):
                if video_file.suffix.lower() in video_extensions:
                    computed = self.compute(str(video_file))
                    if computed == fingerprint:
                        logger.info(f"Video gefunden: {video_file}")
                        return str(video_file)

        return None


# Perceptual Hash (pHash) fuer Aehnlichkeitserkennung
class PerceptualHash:
    """Berechnet Perceptual Hashes fuer Frames/Videos."""

    def __init__(self, hash_size: int = 16):
        """
        Args:
            hash_size: Groesse des Hashes (default 16 = 256 bit)
        """
        self.hash_size = hash_size
        self._imagehash_available = None

    def _check_imagehash(self) -> bool:
        """Prueft ob imagehash installiert ist."""
        if self._imagehash_available is None:
            try:
                import imagehash

                self._imagehash_available = True
            except ImportError:
                logger.warning("imagehash nicht installiert - pHash nicht verfuegbar")
                self._imagehash_available = False
        return self._imagehash_available

    def compute_phash(self, frame: np.ndarray) -> str | None:
        """
        Berechnet Perceptual Hash fuer einen Frame.

        Args:
            frame: OpenCV Frame (BGR)

        Returns:
            Hex-String des Hashes oder None
        """
        if not self._check_imagehash():
            return None

        try:
            import imagehash
            from PIL import Image

            # BGR zu RGB konvertieren
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # pHash berechnen
            phash = imagehash.phash(pil_img, hash_size=self.hash_size)
            return str(phash)

        except Exception as e:
            logger.error(f"pHash Berechnung fehlgeschlagen: {e}")
            return None

    def compute_dhash(self, frame: np.ndarray) -> str | None:
        """Berechnet Difference Hash."""
        if not self._check_imagehash():
            return None

        try:
            import imagehash
            from PIL import Image

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            dhash = imagehash.dhash(pil_img, hash_size=self.hash_size)
            return str(dhash)

        except Exception as e:
            logger.error(f"dHash Berechnung fehlgeschlagen: {e}")
            return None

    def compute_ahash(self, frame: np.ndarray) -> str | None:
        """Berechnet Average Hash."""
        if not self._check_imagehash():
            return None

        try:
            import imagehash
            from PIL import Image

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            ahash = imagehash.average_hash(pil_img, hash_size=self.hash_size)
            return str(ahash)

        except Exception as e:
            logger.error(f"aHash Berechnung fehlgeschlagen: {e}")
            return None

    def compute_all_hashes(self, frame: np.ndarray) -> dict:
        """
        Berechnet alle Hash-Typen fuer einen Frame.

        Returns:
            Dict mit phash, dhash, ahash
        """
        return {
            "phash": self.compute_phash(frame),
            "dhash": self.compute_dhash(frame),
            "ahash": self.compute_ahash(frame),
        }

    def compute_for_video(self, video_path: str, frame_position: str = "middle") -> dict:
        """
        Berechnet Hashes fuer ein Video (mittlerer Frame).

        Args:
            video_path: Pfad zum Video
            frame_position: 'start', 'middle', 'end'

        Returns:
            Dict mit allen Hashes
        """
        # PERF-02 FIX: Use context manager to ensure VideoCapture is released
        from ..utils.video_utils import open_video

        try:
            with open_video(video_path) as cap:
                if not cap.isOpened():
                    return {"phash": None, "dhash": None, "ahash": None}

                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Frame-Position bestimmen
                if frame_position == "start":
                    pos = 0
                elif frame_position == "end":
                    pos = max(0, frame_count - 1)
                else:  # middle
                    pos = frame_count // 2

                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()

            # VideoCapture automatically released by context manager

            if not ret or frame is None:
                return {"phash": None, "dhash": None, "ahash": None}

            return self.compute_all_hashes(frame)

        except Exception as e:
            logger.error(f"Video-Hash Berechnung fehlgeschlagen: {e}")
            return {"phash": None, "dhash": None, "ahash": None}

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """
        Berechnet Hamming-Distanz zwischen zwei Hashes.

        Args:
            hash1, hash2: Hex-Hash-Strings

        Returns:
            Anzahl unterschiedlicher Bits
        """
        if hash1 is None or hash2 is None:
            return -1

        try:
            import imagehash

            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)
            return h1 - h2
        except Exception:
            # Fallback ohne imagehash
            if len(hash1) != len(hash2):
                return -1

            distance = 0
            for c1, c2 in zip(hash1, hash2):
                b1 = int(c1, 16)
                b2 = int(c2, 16)
                xor = b1 ^ b2
                distance += bin(xor).count("1")
            return distance

    @staticmethod
    def is_similar(hash1: str, hash2: str, threshold: int = 20) -> bool:
        """
        Prueft ob zwei Hashes aehnlich sind.

        Args:
            hash1, hash2: Hex-Hash-Strings
            threshold: Maximale Hamming-Distanz fuer Aehnlichkeit

        Returns:
            True wenn aehnlich
        """
        distance = PerceptualHash.hamming_distance(hash1, hash2)
        if distance < 0:
            return False
        return distance <= threshold
