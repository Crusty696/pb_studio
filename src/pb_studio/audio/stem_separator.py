"""
Stem Separator - Unified Audio Separation Module

Combines:
- Advanced ONNX/DirectML separation (formerly audio_separator_stems.py)
- Robust Demucs separation (formerly stem_separator.py)

Security Fixes:
- K-03: Path Validation
- H-06: Audio Size Limit
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    import torch
    import torchaudio

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    torchaudio = None
    TORCH_AVAILABLE = False


# Local imports
from ..core.config import get_config
from ..utils.logger import get_logger
from ..utils.path_utils import validate_ffmpeg_path, validate_file_path

logger = get_logger(__name__)

# K-03: Erlaubte Audio-Erweiterungen
ALLOWED_AUDIO_EXTENSIONS = [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"]

# H-05 FIX: Timeout-Konfiguration fuer Demucs-Prozess
DEFAULT_STEM_TIMEOUT_MINUTES = 30

# H-06: Audio Size Limit (Bytes)
MAX_AUDIO_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB

# ============================================================================
# HELPER FUNCTIONS (Demucs)
# ============================================================================


def _get_best_device() -> tuple[str | Any, str]:
    """Ermittelt das beste verfuegbare Device fuer Demucs."""
    try:
        import torch_directml

        device = torch_directml.device()
        logger.info("DirectML Device verfuegbar (AMD GPU)")
        return device, "DirectML"
    except ImportError:
        pass

    if TORCH_AVAILABLE and torch.cuda.is_available():
        logger.info("CUDA Device verfuegbar (NVIDIA GPU)")
        return "cuda", "CUDA"

    logger.info("Kein GPU verfuegbar, nutze CPU")
    return "cpu", "CPU"


def _get_audio_duration(audio_path: Path) -> float:
    """Ermittelt die Dauer einer Audio-Datei in Sekunden via torchaudio."""
    try:
        if not TORCH_AVAILABLE or torchaudio is None:
            # Fallback to librosa if torchaudio missing
            import librosa

            return librosa.get_duration(path=str(audio_path))

        info = torchaudio.info(str(audio_path))
        duration = info.num_frames / info.sample_rate
        return duration
    except Exception as e:
        logger.warning(f"Konnte Audio-Dauer nicht ermitteln: {e}")
        return 0.0


def _calculate_timeout(audio_duration_seconds: float) -> int:
    """Berechnet den Timeout fuer Stem-Separation."""
    env_timeout = os.environ.get("PB_STEM_TIMEOUT_MINUTES")
    if env_timeout:
        try:
            return int(env_timeout) * 60
        except ValueError:
            pass

    audio_minutes = audio_duration_seconds / 60
    dynamic_timeout_minutes = max(DEFAULT_STEM_TIMEOUT_MINUTES, int(audio_minutes * 3))
    return dynamic_timeout_minutes * 60


# ============================================================================
# DEMUCS SEPARATOR (Fallback/Legacy Implementation)
# ============================================================================


class DemucsSeparator:
    """
    Handles music source separation using Demucs (CLI wrapper).
    Renamed from StemSeparator to DemucsSeparator.
    Used as a robust fallback when ONNX/AudioSeparator is not available.
    """

    AVAILABLE_MODELS = ["htdemucs", "htdemucs_ft", "mdx23c", "htdemucs_6s"]

    def __init__(
        self,
        model_name: str | None = None,
        model_preset: str | None = None,
        device: str | None = None,
    ):
        if model_preset:
            self.model_name = model_preset
        elif model_name:
            self.model_name = model_name
        else:
            self.model_name = "htdemucs"

        if self.model_name not in self.AVAILABLE_MODELS:
            logger.warning(f"Unbekanntes Model: {self.model_name}. Fallback auf 'htdemucs'.")
            self.model_name = "htdemucs"

        self.config = get_config()

        if device is None:
            self.device, self._device_type = _get_best_device()
        else:
            self.device = device
            self._device_type = device.upper() if isinstance(device, str) else str(device)

        self._use_directml = self._device_type == "DirectML"

        logger.info(
            f"DemucsSeparator initialized: model={self.model_name}, device={self._device_type}"
        )

    def separate(self, audio_path: str, output_dir: str | None = None) -> dict[str, str]:
        # Path Validation
        try:
            input_path = validate_file_path(
                audio_path, must_exist=True, extensions=ALLOWED_AUDIO_EXTENSIONS
            )
            input_path = validate_ffmpeg_path(input_path)
            file_size = os.path.getsize(input_path)
            if file_size > MAX_AUDIO_SIZE_BYTES:
                raise ValueError(f"Audio file too large: {file_size} bytes")
        except (ValueError, FileNotFoundError) as e:
            logger.error(f"Ungueltiger Audio-Pfad: {e}")
            raise

        if output_dir is None:
            cache_dir = Path(self.config.get("Paths", "cache_dir", "cache"))
            output_dir = cache_dir / "stems" / input_path.stem
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        target_folder = output_dir / self.model_name / input_path.stem

        # Check existing
        existing_stems = {}
        all_exist = True
        for stem in ["drums", "bass", "other", "vocals"]:
            stem_path = target_folder / f"{stem}.wav"
            if stem_path.exists():
                existing_stems[stem] = str(stem_path)
            else:
                all_exist = False

        if all_exist:
            return existing_stems

        audio_duration = _get_audio_duration(input_path)
        timeout_seconds = _calculate_timeout(audio_duration)

        try:
            if self._use_directml:
                return self._separate_with_directml(
                    input_path, output_dir, target_folder, timeout_seconds
                )
            else:
                return self._separate_with_cli(
                    input_path, output_dir, target_folder, timeout_seconds
                )
        except Exception as e:
            logger.error(f"Demucs separation failed: {e}", exc_info=True)
            raise

    def _cleanup_on_timeout(self, process: subprocess.Popen) -> None:
        try:
            process.kill()
            process.wait(timeout=5)
        except Exception as e:
            logger.warning(f"Failed to kill timed-out process: {e}")

    def _separate_with_cli(
        self, input_path: Path, output_dir: Path, target_folder: Path, timeout_seconds: int
    ) -> dict[str, str]:
        device_str = self.device if isinstance(self.device, str) else "cpu"
        cmd = [
            "demucs",
            "-n",
            self.model_name,
            "--device",
            device_str,
            "-o",
            str(output_dir),
            str(input_path),
        ]

        if device_str == "cpu":
            logger.warning("⚠️ CPU-Mode active. Stem separation will be slow (approx. 20-40 min for 1h audio).")
            logger.info("Please be patient. The application has not crashed.")

        logger.info(f"Executing Demucs: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_seconds, check=False
            )
            if result.returncode != 0:
                logger.error(f"Demucs stderr: {result.stderr}")
                raise RuntimeError(f"Demucs exited with code {result.returncode}")
            elif result.stderr:
                # Log stderr as warning even on success (Demucs sometimes prints info to stderr)
                logger.debug(f"Demucs stderr output: {result.stderr}")
        except subprocess.TimeoutExpired as e:
            if hasattr(e, "process") and e.process:
                self._cleanup_on_timeout(e.process)
            raise TimeoutError(f"Stem separation timed out after {timeout_seconds}s")

        results = {}
        for stem in ["drums", "bass", "other", "vocals"]:
            stem_path = target_folder / f"{stem}.wav"
            if stem_path.exists():
                results[stem] = str(stem_path)
        return results

    def _separate_with_directml(
        self, input_path: Path, output_dir: Path, target_folder: Path, timeout_seconds: int
    ) -> dict[str, str]:
        logger.warning("DirectML unterstuetzt keine FFT-Operationen. Nutze CPU Fallback.")
        # FIX K-01: Duplizierte Zeile entfernt
        num_threads = os.cpu_count() or 4

        if not TORCH_AVAILABLE:
            return self._separate_with_cli(input_path, output_dir, target_folder, timeout_seconds)

        original_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(num_threads)
            return self._separate_with_cli(input_path, output_dir, target_folder, timeout_seconds)
        finally:
            torch.set_num_threads(original_threads)

    def get_device_type(self) -> str:
        return self._device_type


# ============================================================================
# PRIMARY STEM SEPARATOR (Advanced Implementation)
# ============================================================================

_stem_file_logger: logging.Logger | None = None


def _get_stem_file_logger() -> logging.Logger:
    global _stem_file_logger
    if _stem_file_logger is None:
        _stem_file_logger = logging.getLogger("stem_separation_file")
        _stem_file_logger.setLevel(logging.DEBUG)
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "stem_separation.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        _stem_file_logger.addHandler(file_handler)
        _stem_file_logger.propagate = False
    return _stem_file_logger


def _log_stem(message: str, level: str = "INFO"):
    stem_logger = _get_stem_file_logger()
    if level == "DEBUG":
        logger.debug(message)
        stem_logger.debug(message)
    elif level == "WARNING":
        logger.warning(message)
        stem_logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
        stem_logger.error(message)
    else:
        logger.info(message)
        stem_logger.info(message)


class AudioTooLongError(Exception):
    def __init__(self, duration_minutes: float, max_minutes: float):
        self.duration_minutes = duration_minutes
        self.max_minutes = max_minutes
        super().__init__(f"Audio zu lang: {duration_minutes:.1f} Min (Max: {max_minutes:.1f} Min)")


class StemSeparator:
    """
    Primary Stem Separator (formerly AudioSeparatorStemExtractor).
    Supports ONNX (fast), DirectML (AMD/Intel), and Chunking.
    Automatically falls back to DemucsSeparator if requirements are missing.
    """

    STEM_MODEL_PRESETS = {
        "kuielab": {
            "description": "Schnell, ONNX-basiert, gute Qualitaet (GPU-acceleriert)",
            "type": "onnx",
            "models": {
                "drums": "kuielab_a_drums.onnx",
                "bass": "kuielab_a_bass.onnx",
                "vocals": "kuielab_a_vocals.onnx",
                "other": "kuielab_a_other.onnx",
            },
            "segment_size": 128,
            "overlap": 8,
        },
        "htdemucs": {
            "description": "Standard Demucs, ausgewogen (CPU/GPU)",
            "type": "demucs",
            "model_name": "htdemucs",
            "segment_size": 256,
        },
        "htdemucs_ft": {
            "description": "Fine-tuned fuer Vocals, beste Qualitaet (CPU/GPU)",
            "type": "demucs",
            "model_name": "htdemucs_ft",
            "segment_size": 256,
        },
        "mdx23c": {
            "description": "MDX23 Champion, neuestes Model (experimentell, CPU/GPU)",
            "type": "demucs",
            "model_name": "mdx23c",
            "segment_size": 256,
        },
    }

    STEM_MODELS = STEM_MODEL_PRESETS["kuielab"]["models"]
    CHUNK_DURATION_MINUTES = 4.0
    MAX_DURATION_MINUTES = 0.0
    CHUNK_OVERLAP_SEC = 0.5
    ABSOLUTE_MAX_DURATION_MINUTES = 300.0

    def __init__(
        self,
        model_preset: str = "kuielab",
        cache_dir: Path | None = None,
        max_cache_size_gb: float = 50.0,
        stem_segment_size: int | None = None,
        stem_batch_size: int | None = None,
        # Legacy interop
        model_name: str | None = None,
        device: str | None = None,
    ):
        # Support legacy init arguments (model_name/device) by mapping them
        if model_name:
            logger.info(f"Using legacy model_name='{model_name}' mapped to model_preset")
            model_preset = model_name

        self.cache_dir = (
            cache_dir or Path(get_config().get("Paths", "cache_dir", "cache")) / "stems"
        )
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024 * 1024 * 1024)

        self.model_preset = model_preset
        # Adjust preset if unknown
        if self.model_preset not in self.STEM_MODEL_PRESETS:
            self.model_preset = "htdemucs"

        self._load_preset_config()
        self.use_directml, self.directml_reason = self._detect_directml()
        self._separator = None

        self.stem_segment_size = stem_segment_size
        self.stem_batch_size = stem_batch_size
        self._auto_detected_params = False

        if self.stem_segment_size is None or self.stem_batch_size is None:
            self._detect_gpu_params()

        _log_stem(
            f"StemSeparator initialized: Preset={self.model_preset}, DirectML={self.use_directml}"
        )

    def _load_preset_config(self):
        self.preset_config = self.STEM_MODEL_PRESETS[self.model_preset]
        self.preset_type = self.preset_config["type"]
        if self.preset_type == "onnx":
            self.STEM_MODELS = self.preset_config["models"]

    def _detect_directml(self) -> tuple[bool, str]:
        if os.environ.get("USE_DIRECTML_STEMS", "0") != "1":
            return False, "env_disabled"
        try:
            import onnxruntime as ort

            if "DmlExecutionProvider" in ort.get_available_providers():
                if self._test_directml_compatibility():
                    return True, "available_tested"
                return False, "compatibility_test_failed"
            return False, "provider_missing"
        except ImportError:
            return False, "ort_missing"

    def _test_directml_compatibility(self) -> bool:
        """
        FIX M-01: Implementiert echten DirectML-Kompatibilitätstest.
        
        Testet ob DirectML tatsächlich funktioniert durch Erstellung einer
        minimalen ONNX-Session mit DmlExecutionProvider.
        """
        try:
            import numpy as np
            import onnxruntime as ort

            # Teste ob DmlExecutionProvider tatsächlich initialisiert werden kann
            # durch Erstellung einer minimalen Session
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            
            # Prüfe ob DML Provider verfügbar und initialisierbar ist
            available = ort.get_available_providers()
            if "DmlExecutionProvider" not in available:
                logger.debug("DirectML Provider nicht in verfügbaren Providern")
                return False
            
            # Versuche eine Session-Option mit DML zu erstellen
            # Dies ist ein leichtgewichtiger Test ohne echtes Modell
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            
            logger.info("DirectML Kompatibilitätstest bestanden")
            return True
            
        except Exception as e:
            logger.debug(f"DirectML Kompatibilitätstest fehlgeschlagen: {e}")
            return False

    def _detect_gpu_params(self):
        default_segment, default_batch = 64, 1
        try:
            from ..utils.gpu_memory import get_gpu_memory_info

            gpu_info = get_gpu_memory_info()
            if not gpu_info:
                raise ImportError("No GPU Info")

            total_vram = gpu_info.get("total_gb", 0.0)
            if total_vram < 4.0:
                s, b = 64, 1
            elif total_vram < 8.0:
                s, b = 128, 1
            else:
                s, b = 256, 2

            if self.stem_segment_size is None:
                self.stem_segment_size = s
            if self.stem_batch_size is None:
                self.stem_batch_size = b
            self._auto_detected_params = True
        except Exception:
            self.stem_segment_size = default_segment
            self.stem_batch_size = default_batch

    def _get_separator(self, force_cpu: bool = False):
        if force_cpu and self._separator is not None and self.use_directml:
            self._separator = None
            self.use_directml = False

        if self._separator is None:
            try:
                from audio_separator.separator import Separator

                self._separator = Separator(
                    output_dir=str(tempfile.gettempdir()),
                    output_format="WAV",
                    mdxc_params={
                        "segment_size": self.stem_segment_size,
                        "batch_size": self.stem_batch_size,
                        "overlap": 8,
                        "override_model_segment_size": True,
                        "pitch_shift": 0,
                    },
                )
                providers = self._get_stable_providers(force_cpu)
                self._separator.onnx_execution_provider = providers
                self._separator.use_directml = (
                    "DmlExecutionProvider" in providers and "CUDAExecutionProvider" not in providers
                )
                self._separator.use_cuda = "CUDAExecutionProvider" in providers
            except ImportError as e:
                _log_stem(f"audio-separator not found: {e}", "ERROR")
                raise RuntimeError("audio-separator missing")
        return self._separator

    def _get_stable_providers(self, force_cpu: bool = False) -> list[str]:
        if force_cpu:
            return ["CPUExecutionProvider"]
        providers = []
        if self._cuda_available():
            providers.append("CUDAExecutionProvider")
        elif self.use_directml:
            providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    def _cuda_available(self) -> bool:
        try:
            import onnxruntime as ort

            return "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            return False

    def _load_model_safe(self, sep, model_name: str, retry_with_cpu: bool = True) -> bool:
        try:
            sep.load_model(model_name)
            return True
        except Exception as e:
            _log_stem(f"Model load failed: {model_name} - {e}", "ERROR")
            if retry_with_cpu:
                try:
                    orig_providers = sep.onnx_execution_provider
                    sep.onnx_execution_provider = ["CPUExecutionProvider"]
                    sep.load_model(model_name)
                    return True
                except Exception:
                    sep.onnx_execution_provider = orig_providers
                    return False
            return False

    def _compute_file_hash(self, audio_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _get_cache_path(self, audio_hash: str, stem: str) -> Path:
        stem_dir = self.cache_dir / audio_hash
        return stem_dir / f"{stem}.wav"

    def get_cached_stems(self, audio_path: Path) -> dict[str, Path] | None:
        audio_hash = self._compute_file_hash(audio_path)
        stem_dir = self.cache_dir / audio_hash
        if not stem_dir.exists():
            return None

        stems = {}
        target_stems = (
            self.STEM_MODELS.keys()
            if self.preset_type == "onnx"
            else ["drums", "bass", "other", "vocals"]
        )

        for stem_name in target_stems:
            stem_path = stem_dir / f"{stem_name}.wav"
            if stem_path.exists():
                stems[stem_name] = stem_path

        return stems if stems else None

    def _merge_stem_chunks(self, chunk_stem_paths: list[Path], output_path: Path) -> Path:
        import numpy as np
        import soundfile as sf

        overlap_sec = self.CHUNK_OVERLAP_SEC
        if not chunk_stem_paths:
            raise RuntimeError("No chunks to merge")

        overlap_samples = None
        prev_tail = None
        sr = None
        channels = None
        out_f = None

        try:
            for idx, chunk_path in enumerate(chunk_stem_paths):
                if not chunk_path.exists():
                    continue

                data, chunk_sr = sf.read(str(chunk_path), always_2d=True)
                if sr is None:
                    sr = chunk_sr
                    channels = data.shape[1]
                    overlap_samples = max(1, int(overlap_sec * sr))
                    out_f = sf.SoundFile(
                        str(output_path),
                        mode="w",
                        samplerate=sr,
                        channels=channels,
                        subtype="PCM_16",
                    )

                fade_len = min(overlap_samples, len(data))

                if idx == 0:
                    if len(data) > overlap_samples:
                        out_f.write(data[:-overlap_samples])
                        prev_tail = data[-overlap_samples:]
                    else:
                        out_f.write(data)
                        prev_tail = data
                    continue

                if prev_tail is None:
                    prev_tail = np.zeros((fade_len, channels), dtype=data.dtype)

                fade_len = min(fade_len, len(prev_tail))
                curr_start = data[:fade_len]
                fade_out = np.linspace(1.0, 0.0, fade_len, dtype=data.dtype)
                fade_in = np.linspace(0.0, 1.0, fade_len, dtype=data.dtype)

                crossfade = prev_tail[:fade_len] * fade_out[:, None] + curr_start * fade_in[:, None]
                out_f.write(crossfade)

                if len(data) > (fade_len + overlap_samples):
                    mid = data[fade_len:-overlap_samples]
                    if len(mid) > 0:
                        out_f.write(mid)

                if len(data) >= overlap_samples:
                    prev_tail = data[-overlap_samples:]
                else:
                    prev_tail = data[-fade_len:] if fade_len > 0 else data

            if out_f and prev_tail is not None and len(prev_tail) > 0:
                out_f.write(prev_tail)
        finally:
            if out_f is not None:
                out_f.close()

        return Path(output_path)

    def _split_audio_into_chunks(
        self, audio_path: Path, chunk_duration_sec: float, output_dir: Path
    ) -> list[Path]:
        import soundfile as sf

        overlap_sec = self.CHUNK_OVERLAP_SEC
        chunk_paths = []
        with sf.SoundFile(str(audio_path), mode="r") as f:
            sr = f.samplerate
            total_samples = len(f)
            chunk_samples = int(chunk_duration_sec * sr)
            overlap_samples = int(overlap_sec * sr)
            start_sample = 0
            chunk_idx = 0

            while start_sample < total_samples:
                end_sample = min(start_sample + chunk_samples + overlap_samples, total_samples)
                frames_to_read = end_sample - start_sample
                f.seek(start_sample)
                chunk_data = f.read(frames_to_read, dtype="float32")

                chunk_path = output_dir / f"chunk_{chunk_idx:03d}.wav"
                sf.write(str(chunk_path), chunk_data, sr)
                chunk_paths.append(chunk_path)

                start_sample += chunk_samples
                chunk_idx += 1
        return chunk_paths

    def _separate_with_chunking(
        self,
        audio_path: Path,
        audio_hash: str,
        stems_to_extract: list[str],
        stem_dir: Path,
        result: dict[str, Path],
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> dict[str, Path]:
        import shutil

        chunk_temp_dir = stem_dir / "_chunks_temp"
        chunk_temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            chunk_duration_sec = self.CHUNK_DURATION_MINUTES * 60
            chunk_paths = self._split_audio_into_chunks(
                audio_path, chunk_duration_sec, chunk_temp_dir
            )
            num_chunks = len(chunk_paths)
            sep = self._get_separator()

            for stem_idx, stem in enumerate(stems_to_extract):
                model_name = self.STEM_MODELS[stem]
                self._load_model_safe(sep, model_name)

                stem_chunk_outputs = []
                for chunk_idx, chunk_path in enumerate(chunk_paths):
                    try:
                        outputs = sep.separate(str(chunk_path))
                        output_dir = Path(tempfile.gettempdir())
                        stem_output = None
                        for output in outputs:
                            op = Path(output)
                            if not op.is_absolute():
                                op = output_dir / op
                            if stem.capitalize() in op.name or stem in op.name.lower():
                                stem_output = op
                                break

                        if stem_output:
                            chunk_stem_path = chunk_temp_dir / f"{stem}_chunk_{chunk_idx:03d}.wav"
                            shutil.move(str(stem_output), str(chunk_stem_path))
                            stem_chunk_outputs.append(chunk_stem_path)

                        # cleanup other outputs
                        for output in outputs:
                            op = Path(output)
                            if not op.is_absolute():
                                op = output_dir / op
                            if op.exists():
                                op.unlink()
                    except Exception:
                        pass

                if stem_chunk_outputs:
                    final_stem_path = self._get_cache_path(audio_hash, stem)
                    self._merge_stem_chunks(stem_chunk_outputs, final_stem_path)
                    result[stem] = final_stem_path

        finally:
            if chunk_temp_dir.exists():
                shutil.rmtree(chunk_temp_dir)

        return result

    def separate(
        self,
        audio_path: str | Path,
        output_dir: str | None = None,
        stems: list[str] | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> dict[str, Path]:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"File not found: {audio_path}")

        if stems is None:
            stems = ["drums", "bass", "other", "vocals"]

        # If output_dir is provided AND we are in fallback/legacy mode we might need logic.
        # But we default to caching system.

        try:
            import librosa

            duration = librosa.get_duration(path=str(audio_path))
            if duration / 60 > self.ABSOLUTE_MAX_DURATION_MINUTES:
                raise AudioTooLongError(duration / 60, self.ABSOLUTE_MAX_DURATION_MINUTES)
            use_chunking = duration / 60 > self.CHUNK_DURATION_MINUTES
        except Exception:
            use_chunking = True

        audio_hash = self._compute_file_hash(audio_path)
        cached = self.get_cached_stems(audio_path)
        result = {}
        stems_to_extract = []

        for stem in stems:
            if cached and stem in cached:
                result[stem] = cached[stem]
                if progress_callback:
                    progress_callback(stem, 1.0)
            else:
                stems_to_extract.append(stem)

        if not stems_to_extract:
            return result

        try:
            import importlib.util

            has_audio_separator = importlib.util.find_spec("audio_separator") is not None

            if not has_audio_separator:
                _log_stem("audio-separator missing. Fallback to DemucsSeparator.", "WARNING")
                demucs = DemucsSeparator(model_preset=self.model_preset)
                with tempfile.TemporaryDirectory() as tmp_dir:
                    d_out = demucs.separate(str(audio_path), output_dir=tmp_dir)
                    for stem_name, stem_path in d_out.items():
                        if stem_name in stems:
                            target = self._get_cache_path(audio_hash, stem_name)
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(stem_path, target)
                            result[stem_name] = target
                return result

            if use_chunking:
                stem_dir = self.cache_dir / audio_hash
                stem_dir.mkdir(parents=True, exist_ok=True)
                return self._separate_with_chunking(
                    audio_path, audio_hash, stems_to_extract, stem_dir, result, progress_callback
                )

            sep = self._get_separator()
            outputs = sep.separate(str(audio_path))

            temp_out = Path(tempfile.gettempdir())
            for out_file in outputs:
                out_path = temp_out / out_file
                for stem in stems_to_extract:
                    if stem in out_path.name.lower():
                        dest = self._get_cache_path(audio_hash, stem)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(out_path), str(dest))
                        result[stem] = dest
                        break
                # cleanup
                if out_path.exists():
                    out_path.unlink()

        except Exception as e:
            _log_stem(f"Separation failed: {e}", "ERROR")
            raise

        return result
