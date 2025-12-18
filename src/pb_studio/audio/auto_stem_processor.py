"""
Automatic Stem Processor für PB Studio.

Erstellt automatisch Stems bei Audio-Import im Background.
Integriert in den Audio-Analysis Workflow für nahtlose User Experience.

FEATURES:
- Automatische Stem-Erstellung bei Audio-Import
- Individual Stem-Storage (drums.wav, bass.wav, vocals.wav, other.wav)
- Background Processing ohne UI-Blocking
- GPU-beschleunigt (NVIDIA/AMD)
- Intelligent Model-Selection basierend auf Hardware
- Cached Results (keine doppelte Verarbeitung)

WORKFLOW:
Audio Import → BPM/Beatgrid Analysis → Automatic Stem Creation → Individual Storage

MODEL-STRATEGIE:
- GPU verfügbar: KUIELAB (ONNX) für Speed
- Nur CPU: HTDEMUCS für Balance
- Production Mode: HTDEMUCS_FT für beste Qualität
"""

import logging
from pathlib import Path
from typing import Dict, Optional

from ..core.config import get_config
from ..utils.logger import get_logger
from .stem_separator import DemucsSeparator, StemSeparator

logger = get_logger(__name__)


class AutoStemProcessor:
    """
    Automatische Stem-Verarbeitung für Audio-Import.

    Erstellt beim Audio-Import automatisch alle 4 Stem-Spuren:
    - drums.wav
    - bass.wav
    - vocals.wav
    - other.wav

    Nutzt intelligente Model-Auswahl basierend auf verfügbarer Hardware.
    """

    # Model-Auswahl basierend auf Hardware
    SPEED_MODEL = "kuielab"  # Für GPU - schnellste Option
    BALANCED_MODEL = "htdemucs"  # Für CPU - ausgewogen
    QUALITY_MODEL = "htdemucs_ft"  # Für Production - beste Qualität

    def __init__(self, quality_mode: str = "auto"):
        """
        Initialize AutoStemProcessor.

        Args:
            quality_mode: "speed", "balanced", "quality", or "auto"
                         "auto" wählt basierend auf Hardware
        """
        self.config = get_config()
        self.quality_mode = quality_mode
        self._model_name = None
        self._use_onnx = None

        # Cache-Directory für Stems
        self.cache_dir = Path(self.config.get("Paths", "cache_dir", "cache")) / "stems"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _select_optimal_model(self) -> tuple[str, bool]:
        """
        Wähle optimales Model basierend auf Hardware und Quality-Mode.

        Returns:
            Tuple (model_name, use_onnx_extractor)
        """
        if self._model_name and self._use_onnx is not None:
            return self._model_name, self._use_onnx

        # GPU-Detection für Model-Auswahl
        try:
            import torch

            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False

        try:
            import torch_directml

            has_directml = True
        except ImportError:
            has_directml = False

        # Intelligente Model-Auswahl
        if self.quality_mode == "speed":
            model, use_onnx = self.SPEED_MODEL, True
        elif self.quality_mode == "quality":
            model, use_onnx = self.QUALITY_MODEL, False
        elif self.quality_mode == "balanced":
            model, use_onnx = self.BALANCED_MODEL, False
        else:  # auto
            if has_cuda or has_directml:
                # GPU verfügbar: Nutze schnelle ONNX-Models
                model, use_onnx = self.SPEED_MODEL, True
                logger.info("GPU detected: Using KUIELAB (ONNX) for speed")
            else:
                # Nur CPU: Nutze ausgewogenes Demucs
                model, use_onnx = self.BALANCED_MODEL, False
                logger.info("CPU-only: Using HTDEMUCS for balance")

        # Cache-Result
        self._model_name = model
        self._use_onnx = use_onnx

        return model, use_onnx

    def stems_exist(self, audio_path: str) -> tuple[bool, dict[str, str]]:
        """
        Prüfe ob Stems für Audio-Datei bereits existieren.

        Args:
            audio_path: Pfad zur Audio-Datei

        Returns:
            Tuple (all_exist, existing_stems_dict)
        """
        audio_file = Path(audio_path)
        stem_folder = self.cache_dir / audio_file.stem

        stems = {}
        all_exist = True

        for stem_name in ["drums", "bass", "vocals", "other"]:
            stem_path = stem_folder / f"{stem_name}.wav"
            if stem_path.exists() and stem_path.stat().st_size > 0:
                stems[stem_name] = str(stem_path)
            else:
                all_exist = False

        return all_exist, stems

    def create_stems(self, audio_path: str, progress_callback=None) -> dict[str, str]:
        """
        Erstelle Stems für Audio-Datei.

        Args:
            audio_path: Pfad zur Audio-Datei
            progress_callback: Optional callback für Progress Updates

        Returns:
            Dict mit Stem-Namen und Pfaden: {"drums": "path/drums.wav", ...}

        Raises:
            Exception: Wenn Stem-Creation fehlschlägt
        """
        # Check if stems already exist
        all_exist, existing_stems = self.stems_exist(audio_path)
        if all_exist:
            logger.info(f"Stems already exist for {Path(audio_path).name}")
            return existing_stems

        if progress_callback:
            progress_callback(5, "Selecting optimal model...")

        # Select best model for hardware
        model_name, use_onnx = self._select_optimal_model()

        if progress_callback:
            progress_callback(10, f"Creating stems with {model_name}...")

        try:
            if use_onnx:
                # Use ONNX-based StemSeparator (fast GPU)
                extractor = StemSeparator(model_preset=model_name)
                # Note: StemSeparator.separate params adapted to match usage
                stems = extractor.separate(audio_path, stems=["drums", "bass", "vocals", "other"])
                # Convert Dict[str, Path] to Dict[str, str]
                stems = {k: str(v) for k, v in stems.items()}
            else:
                # Use Demucs-based DemucsSeparator (high quality)
                separator = DemucsSeparator(model_preset=model_name)
                # DemucsSeparator returns Dict[str, str]
                stems = separator.separate(audio_path, output_dir=str(self.cache_dir))

            if progress_callback:
                progress_callback(90, "Verifying stem quality...")

            # Verify all stems were created successfully
            missing_stems = []
            for stem_name in ["drums", "bass", "vocals", "other"]:
                if stem_name not in stems or not Path(stems[stem_name]).exists():
                    missing_stems.append(stem_name)

            if missing_stems:
                raise Exception(f"Failed to create stems: {missing_stems}")

            if progress_callback:
                progress_callback(100, "Stems created successfully!")

            logger.info(f"Successfully created {len(stems)} stems for {Path(audio_path).name}")
            return stems

        except Exception as e:
            logger.error(f"Stem creation failed for {audio_path}: {e}")
            raise

    def get_model_info(self) -> dict[str, str]:
        """
        Gibt Informationen über das aktuell gewählte Model zurück.

        Returns:
            Dict mit Model-Info
        """
        model_name, use_onnx = self._select_optimal_model()

        model_type = "ONNX (GPU-accelerated)" if use_onnx else "Demucs (High-quality)"

        return {
            "model_name": model_name,
            "model_type": model_type,
            "quality_mode": self.quality_mode,
            "cache_dir": str(self.cache_dir),
        }


# Singleton Instance für App-weite Verwendung
_auto_stem_processor = None


def get_auto_stem_processor(quality_mode: str = "auto") -> AutoStemProcessor:
    """
    Holt Singleton-Instance vom AutoStemProcessor.

    Args:
        quality_mode: Quality-Mode für Model-Auswahl

    Returns:
        AutoStemProcessor Instance
    """
    global _auto_stem_processor

    if _auto_stem_processor is None or _auto_stem_processor.quality_mode != quality_mode:
        _auto_stem_processor = AutoStemProcessor(quality_mode)

    return _auto_stem_processor
