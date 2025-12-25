"""
GPU Memory Management Utilities for PB_studio.

Provides context managers and utilities for safe GPU memory handling.
Prevents memory leaks and OOM errors during ML/AI operations.

Usage:
    from pb_studio.utils.gpu_memory import torch_device_context, clear_gpu_memory

    # Context manager for automatic cleanup
    with torch_device_context('cuda'):
        model = MyModel().to('cuda')
        output = model(input)
    # GPU memory is automatically freed here

    # Manual cleanup
    clear_gpu_memory()

    # Check available memory
    if check_gpu_memory_available(4.0):
        # Load model that needs 4GB
        pass
"""

import logging
import os
import platform
from contextlib import contextmanager
from typing import Any

# Local imports
from ..core.config import get_config

# Konfigurierbare GPU Memory Reserve (Standard: 20%)
# Kann überschrieben werden durch:
# - Environment Variable: PB_GPU_MEMORY_RESERVE (z.B. 0.15 für 15%)
# - Config-Datei: [Hardware] gpu_reserve = 0.15
DEFAULT_GPU_MEMORY_RESERVE = 0.2  # 20% Reserve

# Empfohlene Werte:
# - 0.10-0.15: Große GPUs (16GB+), stabiles System
# - 0.20: Standard (8-16GB), gute Balance
# - 0.30-0.40: Kleine GPUs (<8GB), sicherer Betrieb

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

try:
    import wmi

    WMI_AVAILABLE = True
except ImportError:
    WMI_AVAILABLE = False
    wmi = None  # type: ignore

logger = logging.getLogger(__name__)

# Bekannte AMD GPUs mit VRAM (Fallback wenn WMI versagt)
# Format: "modellname_keyword": VRAM_in_bytes
KNOWN_AMD_VRAM = {
    "7900 xtx": 24 * 1024**3,
    "7900 xt": 20 * 1024**3,
    "7900 gre": 16 * 1024**3,
    "7800 xt": 16 * 1024**3,
    "7700 xt": 12 * 1024**3,
    "7600": 8 * 1024**3,
    "6950 xt": 16 * 1024**3,
    "6900 xt": 16 * 1024**3,
    "6800 xt": 16 * 1024**3,
    "6800": 16 * 1024**3,
    "6750 xt": 12 * 1024**3,
    "6700 xt": 12 * 1024**3,
    "6650 xt": 8 * 1024**3,
    "6600 xt": 8 * 1024**3,
    "6600": 8 * 1024**3,
}


def get_best_directml_device() -> tuple[Any, int, str] | None:
    """
    Findet das beste DirectML Device (dedizierte GPU bevorzugt).

    Strategie:
    1. Iteriere alle DirectML Devices
    2. Bevorzuge dedizierte GPUs (RX, enthält "XT" oder VRAM in Lookup)
    3. Fallback: Device mit höchstem geschätzten VRAM

    Returns:
        Tuple (device, device_index, device_name) oder None wenn nicht verfügbar
    """
    try:
        import torch_directml

        if not torch_directml.is_available():
            return None

        device_count = torch_directml.device_count()
        if device_count == 0:
            return None

        best_device = None
        best_index = 0
        best_name = ""
        best_score = -1  # Score: höher = besser

        for i in range(device_count):
            name = torch_directml.device_name(i)
            name_lower = name.lower() if name else ""

            # Score berechnen: Dedizierte GPUs bekommen höheren Score
            score = 0

            # RX-Karten sind dediziert (RX 7800 XT, RX 6800 etc.)
            if " rx " in name_lower or name_lower.startswith("rx "):
                score += 1000

            # XT-Suffix = höhere Leistungsklasse
            if " xt" in name_lower:
                score += 100

            # VRAM aus Lookup-Tabelle als zusätzlicher Score
            for model_key, vram in KNOWN_AMD_VRAM.items():
                if model_key in name_lower:
                    score += vram // (1024**3)  # GB als Score
                    break

            # Integrierte GPUs ("Graphics" ohne RX) niedrigerer Score
            if "graphics" in name_lower and "rx" not in name_lower:
                score -= 500

            logger.debug(f"DirectML Device {i}: {name} (Score: {score})")

            if score > best_score:
                best_score = score
                best_device = torch_directml.device(i)
                best_index = i
                best_name = name

        if best_device is not None:
            logger.info(f"Bestes DirectML Device gewählt: {best_name} (Index {best_index})")
            return best_device, best_index, best_name

    except ImportError:
        logger.debug("torch_directml nicht installiert")
    except Exception as e:
        logger.warning(f"DirectML Device-Erkennung fehlgeschlagen: {e}")

    return None


def get_memory_reserve() -> float:
    """
    Hole GPU Memory Reserve aus Config oder Environment.

    Die Memory Reserve bestimmt wieviel GPU-Speicher für System/Overhead
    reserviert bleibt (nicht für Batch-Size-Berechnungen verwendet wird).

    Priorität:
    1. Environment Variable: PB_GPU_MEMORY_RESERVE (0.05 - 0.5)
    2. Config-Datei: [Hardware] gpu_reserve
    3. DEFAULT_GPU_MEMORY_RESERVE (0.2 = 20%)

    Returns:
        Memory Reserve als Float (0.05 - 0.5, geclampt)
    """
    # 1. Environment Variable hat höchste Priorität
    env_reserve = os.environ.get("PB_GPU_MEMORY_RESERVE")
    if env_reserve:
        try:
            reserve = float(env_reserve)
            return max(0.05, min(0.5, reserve))
        except ValueError:
            pass

    # 2. Config-Datei
    try:
        config = get_config()
        config_reserve = config.get("Hardware", "gpu_reserve", fallback=None)
        if config_reserve:
            reserve = float(config_reserve)
            return max(0.05, min(0.5, reserve))
    except Exception:
        pass

    # 3. Fallback: Default-Wert
    return DEFAULT_GPU_MEMORY_RESERVE


@contextmanager
def torch_device_context(device: str | None = None):
    """
    Context manager for PyTorch GPU operations with automatic cleanup.

    This context manager ensures that GPU memory is properly cleaned up
    after operations, preventing memory leaks and OOM errors.

    Args:
        device: Device to use ('cuda', 'dml', 'cpu', 'mps')

    Yields:
        None
    """
    if not TORCH_AVAILABLE:
        yield
        return

    if device is None:
        from pb_studio.core.hardware import get_device

        device = get_device()

    try:
        yield
    finally:
        device_str = str(device) if device is not None else ""
        # Cleanup GPU memory
        if device_str.startswith("cuda") and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug("GPU cache cleared for CUDA")
            except Exception as e:
                logger.warning(f"Error clearing GPU cache for CUDA: {e}")
        elif device_str == "dml" or "privateuseone" in device_str:
            try:
                import torch_directml

                torch_directml.empty_cache()
                logger.debug("GPU cache cleared for DirectML")
            except (ImportError, AttributeError):
                logger.warning(
                    "torch_directml not found or empty_cache not available, cannot clear cache."
                )
            except Exception as e:
                logger.warning(f"Error clearing GPU cache for DirectML: {e}")


def get_gpu_memory_info() -> dict | None:
    """
    Get current GPU memory usage.

    Returns:
        Dict with memory info or None if GPU not available
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9

            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "free_gb": total - allocated,
                "utilization_pct": (allocated / total) * 100,
                "provider": "cuda",
            }
        except Exception as e:
            logger.error(f"Error getting GPU memory info: {e}")
            return None

    # CUDA nicht verfuegbar: versuche DirectML (Windows)
    dml_info = get_directml_memory_info()
    if dml_info:
        return dml_info

    return None


def get_directml_memory_info() -> dict | None:
    """
    Liefert eine konservative Schaetzung des verfuegbaren DirectML-GPU-Speichers.
    """
    if platform.system() != "Windows":
        logger.debug("DirectML nur auf Windows verfügbar")
        return None

    device_name = None
    try:
        import torch_directml

        if torch_directml.is_available():
            device_name = torch_directml.device_name(0)
            logger.debug(f"DirectML device gefunden: {device_name}")
    except Exception as e:
        logger.debug(f"torch_directml nicht verfügbar: {e}")
        device_name = None

    controllers = _get_wmi_video_controllers()
    if not controllers:
        logger.warning("WMI-Abfrage fehlgeschlagen, verwende Fallback (4GB)")
        total_gb = 4.0
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "total_gb": total_gb,
            "free_gb": total_gb,
            "utilization_pct": 0.0,
            "provider": "directml",
            "device_name": device_name or "Unknown DirectML Device",
            "fallback": True,
        }

    chosen = _pick_matching_controller(controllers, device_name)
    if not chosen or not chosen.get("memory_bytes"):
        logger.warning("Kein passender GPU-Controller gefunden, verwende Fallback (4GB)")
        total_gb = 4.0
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "total_gb": total_gb,
            "free_gb": total_gb,
            "utilization_pct": 0.0,
            "provider": "directml",
            "device_name": device_name or "Unknown DirectML Device",
            "fallback": True,
        }

    total_gb = chosen["memory_bytes"] / 1e9
    allocated = 0.0  # DirectML liefert keine Live-Daten
    logger.info(f"DirectML GPU erkannt: {chosen.get('name')}, VRAM: {total_gb:.2f} GB")
    return {
        "allocated_gb": allocated,
        "reserved_gb": allocated,
        "total_gb": total_gb,
        "free_gb": total_gb - allocated,
        "utilization_pct": 0.0,
        "provider": "directml",
        "device_name": chosen.get("name"),
        "fallback": False,
    }


def clear_gpu_memory():
    """
    Clear GPU cache manually.
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return

    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("GPU memory cleared manually")
    except Exception as e:
        logger.error(f"Error clearing GPU memory: {e}")


def check_gpu_memory_available(required_gb: float) -> bool:
    """
    Check if enough GPU memory is available.
    """
    info = get_gpu_memory_info()
    if not info:
        return False

    available = info["free_gb"]
    logger.debug(f"GPU memory: {available:.2f} GB free, {required_gb:.2f} GB required")

    return available >= required_gb


def log_gpu_memory_usage(prefix: str = ""):
    """
    Log current GPU memory usage.
    """
    info = get_gpu_memory_info()
    if not info:
        logger.debug(f"{prefix} - GPU not available")
        return

    logger.info(
        f"{prefix} - GPU Memory: {info['allocated_gb']:.2f} GB / {info['total_gb']:.2f} GB "
        f"({info['utilization_pct']:.1f}%)"
    )


def estimate_max_batch_size(model_size_gb: float, sample_size_gb: float) -> int:
    """
    Estimate maximum batch size based on available GPU memory.
    """
    info = get_gpu_memory_info()
    if not info:
        logger.debug("GPU Memory Info nicht verfügbar, verwende Batch-Size 1")
        return 1

    reserve = get_memory_reserve()

    if info.get("provider") == "directml":
        effective_reserve = min(0.5, reserve + 0.1)
    else:
        effective_reserve = reserve

    available = info["total_gb"] * (1 - effective_reserve) - model_size_gb

    if available <= 0:
        return 1

    batch_size = int(available / sample_size_gb)
    return max(1, batch_size)


class GPUMemoryMonitor:
    """
    Context manager for monitoring GPU memory usage.
    """

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_info = None
        self.end_info = None

    def __enter__(self):
        self.start_info = get_gpu_memory_info()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_info = get_gpu_memory_info()
        clear_gpu_memory()
        return False


def _get_wmi_video_controllers() -> list[dict[str, Any]] | None:
    """
    Hole GPU-Infos ueber WMI (Windows-only).
    """
    if platform.system() != "Windows":
        return None

    if not WMI_AVAILABLE:
        return None

    try:
        conn = wmi.WMI()  # type: ignore
        controllers: list[dict[str, Any]] = []
        for gpu in conn.Win32_VideoController():
            name = getattr(gpu, "Name", None)
            adapter_ram = getattr(gpu, "AdapterRAM", None)
            dedicated = getattr(gpu, "DedicatedVideoMemory", None)

            vram = None
            for cand in (dedicated, adapter_ram):
                if cand and cand > 0:
                    vram = int(cand)
                    break

            if not vram and name:
                name_lower = name.lower()
                for model_key, known_vram in KNOWN_AMD_VRAM.items():
                    if model_key in name_lower:
                        vram = known_vram
                        break

            controllers.append({"name": name, "memory_bytes": vram})

        return controllers if controllers else None

    except Exception:
        return None


def _pick_matching_controller(
    controllers: list[dict[str, Any]], preferred_name: str | None
) -> dict[str, Any] | None:
    """
    Waehlt einen passenden GPU-Eintrag. Bevorzugt Namens-Matches, sonst groessten VRAM.
    """
    if not controllers:
        return None

    if preferred_name:
        preferred_lower = preferred_name.lower()
        for gpu in controllers:
            name = (gpu.get("name") or "").lower()
            if preferred_lower in name or name in preferred_lower:
                return gpu

    valid_gpus = [gpu for gpu in controllers if gpu.get("memory_bytes")]
    if valid_gpus:
        return max(valid_gpus, key=lambda x: x.get("memory_bytes", 0))

    return controllers[0]