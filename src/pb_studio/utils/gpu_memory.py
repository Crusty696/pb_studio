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

# Konfigurierbare GPU Memory Reserve (Standard: 20%)
# Kann überschrieben werden durch:
# - Environment Variable: PB_GPU_MEMORY_RESERVE (z.B. 0.15 für 15%)
# - TODO: Integration mit zentralem Config-Manager
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


def get_memory_reserve() -> float:
    """
    Hole GPU Memory Reserve aus Config oder Environment.

    Die Memory Reserve bestimmt wieviel GPU-Speicher für System/Overhead
    reserviert bleibt (nicht für Batch-Size-Berechnungen verwendet wird).

    Priorität:
    1. Environment Variable: PB_GPU_MEMORY_RESERVE (0.05 - 0.5)
    2. TODO: Config-Datei (zentraler Config-Manager)
    3. DEFAULT_GPU_MEMORY_RESERVE (0.2 = 20%)

    Returns:
        Memory Reserve als Float (0.05 - 0.5, geclampt)

    Usage:
        # Setze 15% Reserve via Environment
        # Linux/WSL:  export PB_GPU_MEMORY_RESERVE=0.15
        # Windows:    set PB_GPU_MEMORY_RESERVE=0.15
        # PowerShell: $env:PB_GPU_MEMORY_RESERVE = "0.15"

        reserve = get_memory_reserve()
        # Returns: 0.15
    """
    # Environment Variable hat Priorität
    env_reserve = os.environ.get("PB_GPU_MEMORY_RESERVE")
    if env_reserve:
        try:
            reserve = float(env_reserve)
            # Clamp zwischen 5% und 50%
            clamped = max(0.05, min(0.5, reserve))
            if clamped != reserve:
                logger.warning(
                    f"GPU Memory Reserve {reserve:.2%} außerhalb erlaubtem Bereich "
                    f"(5%-50%), geclampt auf {clamped:.2%}"
                )
            logger.debug(f"GPU Memory Reserve aus Environment: {clamped:.2%}")
            return clamped
        except ValueError:
            logger.warning(
                f"Ungültiger Wert für PB_GPU_MEMORY_RESERVE: '{env_reserve}', "
                f"verwende Standard ({DEFAULT_GPU_MEMORY_RESERVE:.2%})"
            )

    # Fallback: Default-Wert
    return DEFAULT_GPU_MEMORY_RESERVE


@contextmanager
def torch_device_context(device: str | None = None):
    """
    Context manager for PyTorch GPU operations with automatic cleanup.

    This context manager ensures that GPU memory is properly cleaned up
    after operations, preventing memory leaks and OOM errors.

    Usage:
        from pb_studio.core.hardware import get_device
        device = get_device()
        with torch_device_context(device):
            model = MyModel().to(device)
            output = model(input)
        # GPU memory is automatically freed here

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
        Dict with memory info:
        - allocated_gb: Currently allocated memory in GB
        - reserved_gb: Reserved memory in GB
        - total_gb: Total GPU memory in GB
        - free_gb: Free memory in GB
        - utilization_pct: Memory utilization percentage

        Returns None if GPU not available
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

    DirectML/Torch DirectML exposes keine Laufzeit-Memory-APIs, daher nutzen wir:
    - torch_directml.device_name(0) fuer die Namenswahl (falls vorhanden)
    - WMI (Win32_VideoController) fuer die gesamte VRAM-Groesse

    Fallback-Strategie:
    1. Versuche WMI-basierte Abfrage (Win32_VideoController)
    2. Falls WMI fehlschlaegt: Konservative Schaetzung (4GB)

    Returns:
        Dict mit DirectML-Speicherinfo oder None bei Fehler
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
        # Keine harte Abhaengigkeit; WMI-Fallback reicht
        logger.debug(f"torch_directml nicht verfügbar: {e}")
        device_name = None

    controllers = _get_wmi_video_controllers()
    if not controllers:
        logger.warning("WMI-Abfrage fehlgeschlagen, verwende Fallback (4GB)")
        # Fallback: Konservative Schaetzung
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
        # Fallback
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

    This function empties the CUDA cache and synchronizes the device.
    Useful for freeing up memory between operations.

    Usage:
        clear_gpu_memory()
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

    Args:
        required_gb: Required memory in GB

    Returns:
        True if enough memory available, False otherwise

    Usage:
        if check_gpu_memory_available(4.0):
            # Load model that needs 4GB
            model = HeavyModel().to('cuda')
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

    Args:
        prefix: Optional prefix for log message

    Usage:
        log_gpu_memory_usage("After model load")
        # Output: After model load - GPU Memory: 4.23 GB / 24.00 GB (17.6%)
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

    Berücksichtigt:
    - Konfigurierbare Memory Reserve (via get_memory_reserve())
    - Provider-spezifische Overhead-Faktoren (DirectML vs CUDA)
    - Model-Größe und Sample-Größe

    Args:
        model_size_gb: Model size in GB
        sample_size_gb: Single sample size in GB

    Returns:
        Estimated max batch size (at least 1)

    Usage:
        # Standard (20% Reserve)
        max_batch = estimate_max_batch_size(
            model_size_gb=2.0,
            sample_size_gb=0.1
        )
        # Returns: 11 (bei 16GB GPU: 16 * 0.8 - 2.0 = 10.8GB → 10.8 / 0.1 = 108 → int = 11)

        # Mit niedriger Reserve (15%)
        # export PB_GPU_MEMORY_RESERVE=0.15
        max_batch = estimate_max_batch_size(
            model_size_gb=2.0,
            sample_size_gb=0.1
        )
        # Returns: 116 (bei 16GB GPU: 16 * 0.85 - 2.0 = 11.6GB → 11.6 / 0.1 = 116)
    """
    info = get_gpu_memory_info()
    if not info:
        logger.debug("GPU Memory Info nicht verfügbar, verwende Batch-Size 1")
        return 1

    # Hole konfigurierbare Memory Reserve
    reserve = get_memory_reserve()

    # DirectML benötigt zusätzlichen Overhead (10% extra)
    # CUDA kann mit der konfigurierten Reserve arbeiten
    if info.get("provider") == "directml":
        # DirectML: Reserve + 10% zusätzlicher Overhead
        effective_reserve = min(0.5, reserve + 0.1)
        logger.debug(
            f"DirectML: Verwende {effective_reserve:.2%} Reserve (Standard {reserve:.2%} + 10% DirectML-Overhead)"
        )
    else:
        # CUDA/andere: Nur konfigurierte Reserve
        effective_reserve = reserve

    # Verfügbarer Speicher für Batching
    available = info["total_gb"] * (1 - effective_reserve) - model_size_gb

    logger.debug(
        f"GPU Memory: {info['total_gb']:.1f}GB total, "
        f"{effective_reserve * 100:.0f}% reserve, "
        f"{model_size_gb:.1f}GB model, "
        f"{available:.1f}GB available for batching"
    )

    if available <= 0:
        logger.warning(
            f"Nicht genug GPU Memory für Modell ({model_size_gb:.1f}GB benötigt, "
            f"{info['total_gb'] * (1 - effective_reserve):.1f}GB verfügbar nach Reserve)"
        )
        return 1

    batch_size = int(available / sample_size_gb)
    final_batch_size = max(1, batch_size)

    logger.debug(
        f"Geschätzte Max Batch-Size: {final_batch_size} "
        f"({available:.1f}GB / {sample_size_gb:.3f}GB per sample)"
    )

    return final_batch_size


class GPUMemoryMonitor:
    """
    Context manager for monitoring GPU memory usage.

    Usage:
        with GPUMemoryMonitor("Model training"):
            train_model()
        # Output:
        # Model training - Start: 2.34 GB (9.8%)
        # Model training - End: 4.56 GB (19.0%)
        # Model training - Delta: +2.22 GB
    """

    def __init__(self, operation_name: str):
        """
        Initialize memory monitor.

        Args:
            operation_name: Name of the operation being monitored
        """
        self.operation_name = operation_name
        self.start_info = None
        self.end_info = None

    def __enter__(self):
        """Record start memory usage."""
        self.start_info = get_gpu_memory_info()
        if self.start_info:
            logger.info(
                f"{self.operation_name} - Start: {self.start_info['allocated_gb']:.2f} GB "
                f"({self.start_info['utilization_pct']:.1f}%)"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Record end memory usage and report delta."""
        self.end_info = get_gpu_memory_info()

        if self.start_info and self.end_info:
            delta = self.end_info["allocated_gb"] - self.start_info["allocated_gb"]
            logger.info(
                f"{self.operation_name} - End: {self.end_info['allocated_gb']:.2f} GB "
                f"({self.end_info['utilization_pct']:.1f}%)"
            )
            logger.info(f"{self.operation_name} - Delta: {delta:+.2f} GB")

        # Cleanup
        clear_gpu_memory()

        return False  # Don't suppress exceptions


def _get_wmi_video_controllers() -> list[dict[str, Any]] | None:
    """
    Hole GPU-Infos ueber WMI (Windows-only).

    Returns:
        Liste von GPU-Controllern mit Name und VRAM, oder None bei Fehler
    """
    if platform.system() != "Windows":
        logger.debug("WMI nur auf Windows verfügbar")
        return None

    if not WMI_AVAILABLE:
        logger.warning("WMI-Modul nicht installiert (pip install WMI)")
        return None

    try:
        conn = wmi.WMI()  # type: ignore
        controllers: list[dict[str, Any]] = []
        for gpu in conn.Win32_VideoController():
            name = getattr(gpu, "Name", None)
            adapter_ram = getattr(gpu, "AdapterRAM", None)
            dedicated = getattr(gpu, "DedicatedVideoMemory", None)

            # Versuche DedicatedVideoMemory zuerst, dann AdapterRAM
            vram = None
            for cand in (dedicated, adapter_ram):
                if cand and cand > 0:
                    vram = int(cand)
                    break

            if vram:
                logger.debug(f"WMI GPU gefunden: {name}, VRAM: {vram / 1e9:.2f} GB")
            else:
                logger.debug(f"WMI GPU gefunden: {name}, aber kein VRAM erkannt")

            controllers.append({"name": name, "memory_bytes": vram})

        if not controllers:
            logger.warning("Keine GPUs via WMI gefunden")

        return controllers if controllers else None

    except Exception as exc:  # pragma: no cover - WMI optional
        logger.warning(f"WMI GPU-Abfrage fehlgeschlagen: {exc}")
        return None


def _pick_matching_controller(
    controllers: list[dict[str, Any]], preferred_name: str | None
) -> dict[str, Any] | None:
    """
    Waehlt einen passenden GPU-Eintrag. Bevorzugt Namens-Matches, sonst groessten VRAM.

    Strategie:
    1. Wenn preferred_name gesetzt: Suche Namens-Match
    2. Sonst: Waehle GPU mit groesstem VRAM
    3. Fallback: Erste GPU in Liste

    Args:
        controllers: Liste von GPU-Controllern
        preferred_name: Bevorzugter GPU-Name (z.B. von torch_directml)

    Returns:
        Passender GPU-Eintrag oder None
    """
    if not controllers:
        return None

    # 1. Versuche Namens-Match
    if preferred_name:
        preferred_lower = preferred_name.lower()
        for gpu in controllers:
            name = (gpu.get("name") or "").lower()
            if preferred_lower in name or name in preferred_lower:
                logger.debug(f"GPU-Match gefunden: {gpu.get('name')}")
                return gpu

    # 2. Waehle GPU mit groesstem VRAM
    valid_gpus = [gpu for gpu in controllers if gpu.get("memory_bytes")]
    if valid_gpus:
        largest = max(valid_gpus, key=lambda x: x.get("memory_bytes", 0))
        logger.debug(f"Groesste GPU gewaehlt: {largest.get('name')}")
        return largest

    # 3. Fallback: Erste GPU
    logger.debug("Fallback: Erste GPU in Liste")
    return controllers[0]
