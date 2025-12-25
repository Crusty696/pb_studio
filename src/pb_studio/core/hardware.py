"""
Hardware-Erkennung für PB_studio

Erkennt verfügbare Hardware (CPU, NVIDIA GPU mit CUDA, AMD GPU, Intel iGPU) und
konfiguriert die Anwendung entsprechend.

Unterstützte Hardware:
- NVIDIA GPU: Volle CUDA-Unterstützung (PyTorch + NVENC)
- AMD GPU: ROCm auf Linux (PyTorch), DirectML auf Windows (ONNX), AMF-Rendering
- Intel iGPU: DirectML auf Windows (ONNX), QSV-Rendering
- Apple Silicon: MPS (Metal Performance Shaders)
- CPU: Vollständiger Fallback für alle Funktionen

Auto-Detection Features:
- PyTorch Device Selection (cuda/mps/cpu)
- ONNX Runtime Execution Providers
- FFmpeg Hardware Encoder Selection
- Memory-optimized Settings
"""

import os
import platform
from typing import Any

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

from ..utils.logger import get_logger

logger = get_logger()


class HardwareInfo:
    """Container für Hardware-Informationen mit erweiterten Features."""

    def __init__(self):
        # Platform Info
        self.platform = platform.system()
        self.cpu_count = os.cpu_count() or 1

        # GPU Detection Flags
        self.has_nvidia_cuda = False
        self.has_amd_gpu = False
        self.has_intel_igpu = False
        self.has_rocm = False  # AMD ROCm (Linux only)
        self.has_directml = False  # DirectML (Windows AI acceleration)

        # Device Info
        self.compute_device = "cpu"  # Legacy: 'cuda' or 'cpu'
        self.pytorch_device: str | None = None  # 'cuda', 'mps', 'cpu'
        self.gpu_name: str | None = None
        self.cuda_version: str | None = None
        self.gpu_memory_gb: float | None = None

        # ONNX Runtime
        self.onnx_providers: list[str] = []

        # FFmpeg Encoder (set by VideoRenderer)
        self.ffmpeg_encoder: str | None = None

        # Optimized Settings
        self.recommended_batch_size: int = 1
        self.recommended_num_workers: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for logging/debugging."""
        return {
            "platform": self.platform,
            "cpu_count": self.cpu_count,
            "compute_device": self.compute_device,
            "pytorch_device": self.pytorch_device,
            "gpu_name": self.gpu_name,
            "has_nvidia_cuda": self.has_nvidia_cuda,
            "has_amd_gpu": self.has_amd_gpu,
            "has_intel_igpu": self.has_intel_igpu,
            "has_rocm": self.has_rocm,
            "has_directml": self.has_directml,
            "onnx_providers": self.onnx_providers,
            "ffmpeg_encoder": self.ffmpeg_encoder,
            "gpu_memory_gb": self.gpu_memory_gb,
            "recommended_batch_size": self.recommended_batch_size,
            "recommended_num_workers": self.recommended_num_workers,
        }

    def __repr__(self) -> str:
        return (
            f"HardwareInfo(platform={self.platform}, "
            f"cpu_count={self.cpu_count}, "
            f"pytorch_device={self.pytorch_device}, "
            f"gpu_name={self.gpu_name})"
        )


def detect_nvidia_cuda() -> tuple[bool, str | None, str | None]:
    """
    Erkennt NVIDIA GPU mit CUDA-Unterstützung.

    Returns:
        Tuple (has_cuda, gpu_name, cuda_version)
    """
    if not TORCH_AVAILABLE:
        logger.info("PyTorch nicht verfügbar - kann CUDA nicht prüfen")
        return False, None, None

    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            logger.info(f"NVIDIA GPU mit CUDA gefunden: {gpu_name} (CUDA {cuda_version})")
            return True, gpu_name, cuda_version
        else:
            logger.info("Keine NVIDIA GPU mit CUDA verfügbar")
            return False, None, None
    except Exception as e:
        logger.error(f"Fehler bei CUDA-Erkennung: {e}")
        return False, None, None


def detect_amd_gpu_windows() -> tuple[bool, str | None]:
    """
    Erkennt AMD GPU unter Windows via WMI.

    Returns:
        Tuple (has_amd_gpu, gpu_name)
    """
    if platform.system() != "Windows":
        return False, None

    if not WMI_AVAILABLE:
        logger.warning("WMI-Bibliothek nicht verfügbar - kann AMD GPU nicht erkennen")
        return False, None

    try:
        c = wmi.WMI()  # type: ignore
        amd_gpus = [
            gpu for gpu in c.Win32_VideoController() if "AMD" in gpu.Name or "Radeon" in gpu.Name
        ]

        if amd_gpus:
            gpu_name = amd_gpus[0].Name
            logger.info(f"AMD GPU gefunden: {gpu_name}")
            logger.info(
                "AMD GPU erkannt - verwende CPU-Fallback für PyTorch "
                "(ROCm unter Windows nicht unterstützt)"
            )
            return True, gpu_name
        else:
            logger.info("Keine AMD GPU gefunden")
            return False, None

    except Exception as e:
        logger.error(f"Fehler bei AMD GPU-Erkennung via WMI: {e}")
        return False, None


def detect_intel_igpu_windows() -> dict[str, Any] | None:
    """
    Erkennt Intel iGPU unter Windows via WMI (Opt-In ueber PB_ENABLE_INTEL_IGPU=1).

    Unterscheidet zwischen:
    - Intel Arc (diskrete GPU)
    - Intel UHD/Iris/HD Graphics (integrierte GPU)

    Returns:
        Dict mit 'type' (discrete/integrated), 'name', 'vram_gb' oder None
    """
    if platform.system() != "Windows":
        logger.debug("Intel iGPU Detection nur unter Windows verfuegbar")
        return None

    # Opt-In via Environment Variable
    if os.environ.get("PB_ENABLE_INTEL_IGPU", "0").lower() not in {"1", "true", "yes", "on"}:
        logger.debug(
            "Intel iGPU Detection deaktiviert (setze PB_ENABLE_INTEL_IGPU=1 zum Aktivieren)"
        )
        return None

    if not WMI_AVAILABLE:
        logger.warning("WMI-Bibliothek nicht verfuegbar - Intel iGPU kann nicht erkannt werden")
        return None

    try:
        conn = wmi.WMI()  # type: ignore
        intel_gpus = [
            gpu
            for gpu in conn.Win32_VideoController()
            if gpu.Name and "intel" in gpu.Name.lower() and "microsoft" not in gpu.Name.lower()
        ]

        if not intel_gpus:
            logger.debug("Keine Intel GPU via WMI gefunden")
            return None

        gpu = intel_gpus[0]
        gpu_name = gpu.Name
        name_lower = gpu_name.lower()

        # VRAM auslesen (in Bytes, konvertieren zu GB)
        vram_gb = None
        if hasattr(gpu, "AdapterRAM") and gpu.AdapterRAM:
            try:
                vram_gb = float(gpu.AdapterRAM) / (1024**3)
            except (ValueError, TypeError):
                pass

        # Unterscheidung: Arc (diskret) vs UHD/Iris/HD (integriert)
        if "arc" in name_lower:
            gpu_type = "discrete"
            logger.info(f"Intel Arc (diskrete GPU) erkannt: {gpu_name}")
            if vram_gb:
                logger.info(f"  VRAM: {vram_gb:.2f} GB")
        elif any(keyword in name_lower for keyword in ["uhd", "iris", "hd graphics"]):
            gpu_type = "integrated"
            logger.info(f"Intel iGPU (integriert) erkannt: {gpu_name}")
            if vram_gb:
                logger.info(f"  Shared Memory: {vram_gb:.2f} GB")
        else:
            # Unbekannter Intel GPU Typ - konservativ als integriert behandeln
            gpu_type = "integrated"
            logger.info(f"Intel GPU erkannt (Typ unbekannt): {gpu_name}")
            logger.debug("  Behandle als integriert (Arc-Keyword nicht gefunden)")

        return {"type": gpu_type, "name": gpu_name, "vram_gb": vram_gb}

    except ImportError:
        logger.warning("WMI-Import fehlgeschlagen - Intel iGPU Detection nicht moeglich")
        return None
    except Exception as exc:
        logger.error(f"Fehler bei Intel iGPU-Erkennung via WMI: {exc}")
        return None


def detect_and_get_torch_device() -> tuple[str, float | None]:
    """
    Erkennt optimales PyTorch Device.
    Prioritaet: CUDA > DirectML > MPS > CPU

    Returns:
        Tuple (device_string, gpu_memory_gb)
    """
    if not TORCH_AVAILABLE:
        logger.info("PyTorch nicht verfuegbar - verwende CPU")
        return "cpu", None

    try:
        # 1. Check for CUDA (NVIDIA or AMD ROCm)
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

            if torch.version.cuda:
                logger.info(f"NVIDIA GPU erkannt: {gpu_name} ({gpu_memory:.2f} GB)")
                torch.backends.cudnn.benchmark = True
            else:
                logger.info(f"AMD GPU mit ROCm erkannt: {gpu_name} ({gpu_memory:.2f} GB)")
            return device, gpu_memory

        # 2. Check for DirectML (Windows)
        directml_device = detect_directml_device()
        if directml_device:
            logger.info("DirectML-Geraet fuer PyTorch gefunden (privateuseone).")
            return str(directml_device), None

        # 3. Check for Apple Metal (MPS)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple Metal (MPS) erkannt")
            return "mps", None

        # 4. Fallback to CPU
        else:
            logger.info("Keine GPU fuer PyTorch verfuegbar - verwende CPU")
            return "cpu", None

    except Exception as e:
        logger.error(f"Fehler bei PyTorch Device Detection: {e}")
        return "cpu", None


def detect_onnx_providers() -> list[str]:
    """
    Erkennt verfügbare ONNX Runtime Execution Providers.

    Priority (best to worst):
    1. TensorRT (NVIDIA optimized)
    2. CUDA (NVIDIA)
    3. ROCm (AMD Linux)
    4. MIGraphX (AMD optimized)
    5. DirectML (Windows AMD/Intel/NVIDIA)
    6. OpenVINO (Intel optimized)
    7. CPU (fallback)

    Returns:
        List of available providers in priority order
    """
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()

        # Priority order
        preferred_order = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "MIGraphXExecutionProvider",
            "DmlExecutionProvider",  # DirectML for Windows
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ]

        # Filter and sort by priority
        providers = [p for p in preferred_order if p in available]

        if providers:
            logger.info(f"ONNX Runtime providers: {', '.join(providers[:2])}")
        else:
            logger.debug("Keine ONNX Runtime providers verfügbar")

        return providers

    except ImportError:
        logger.debug("ONNX Runtime nicht installiert")
        return []
    except OSError as e:
        # DLL load failure (e.g., missing Visual C++ Redistributable)
        logger.error(f"ONNX Runtime DLL-Fehler: {e}")
        return []
    except Exception as e:
        logger.error(f"Fehler bei ONNX Provider Detection: {e}")
        return []


def get_recommended_settings(hw_info: "HardwareInfo") -> dict[str, Any]:
    """
    Gibt hardware-optimierte Settings zurück.

    Returns:
        Dict with recommended settings for various operations
    """
    settings = {
        "num_workers": min(hw_info.cpu_count, 8),  # DataLoader workers
        "batch_size": 1,
        "pin_memory": False,
        "enable_amp": False,  # Automatic Mixed Precision
        "enable_compile": False,  # torch.compile()
    }

    device_str = str(hw_info.pytorch_device or "")

    # GPU-specific optimizations
    if device_str == "cuda":
        if hw_info.gpu_memory_gb:
            # Batch size based on GPU memory
            if hw_info.gpu_memory_gb >= 16:
                settings["batch_size"] = 8
            elif hw_info.gpu_memory_gb >= 8:
                settings["batch_size"] = 4
            elif hw_info.gpu_memory_gb >= 4:
                settings["batch_size"] = 2

        settings["pin_memory"] = True  # Faster GPU transfer
        settings["enable_amp"] = True  # Mixed precision for speed

        # torch.compile for PyTorch 2.0+ (NVIDIA only, very fast!)
        if TORCH_AVAILABLE and hasattr(torch, "compile") and torch.version.cuda:
            settings["enable_compile"] = True

    # CPU-specific optimizations
    elif device_str.startswith("dml") or "privateuseone" in device_str:
        # DirectML: konservativ starten, optional per VRAM skalieren
        if hw_info.gpu_memory_gb and hw_info.gpu_memory_gb >= 12:
            settings["batch_size"] = 4
        elif hw_info.gpu_memory_gb and hw_info.gpu_memory_gb >= 8:
            settings["batch_size"] = 2
        else:
            settings["batch_size"] = 1
        settings["enable_amp"] = True
    elif device_str == "cpu":
        # More workers for CPU to parallelize
        settings["num_workers"] = hw_info.cpu_count
        settings["batch_size"] = 4  # CPU can handle larger batches

    return settings


def detect_hardware() -> HardwareInfo:
    """
    Führt vollständige Hardware-Erkennung durch.

    Erkennt:
    - PyTorch Device (NVIDIA CUDA / AMD ROCm / Apple MPS / CPU)
    - NVIDIA GPU mit CUDA
    - AMD GPU (Windows WMI, Linux ROCm)
    - Intel iGPU (Windows WMI)
    - ONNX Runtime Execution Providers
    - Optimierte Settings basierend auf Hardware

    Returns:
        HardwareInfo-Objekt mit erkannter Hardware
    """
    logger.info("=" * 60)
    logger.info("Starte Hardware-Erkennung...")
    logger.info("=" * 60)

    hw_info = HardwareInfo()

    # 1. PyTorch Device Detection (NVIDIA/AMD/Apple/CPU)
    pytorch_device, gpu_memory = detect_and_get_torch_device()
    hw_info.pytorch_device = pytorch_device
    hw_info.gpu_memory_gb = gpu_memory

    # 2. NVIDIA CUDA Detection
    has_cuda, gpu_name, cuda_version = detect_nvidia_cuda()
    hw_info.has_nvidia_cuda = has_cuda
    if has_cuda:
        hw_info.compute_device = "cuda"
        hw_info.gpu_name = gpu_name
        hw_info.cuda_version = cuda_version

    # 3. AMD GPU Detection
    if not has_cuda:
        has_amd, amd_name = detect_amd_gpu_windows()
        hw_info.has_amd_gpu = has_amd
        if has_amd:
            hw_info.gpu_name = amd_name
            # Check if ROCm is available (Linux only)
            if platform.system() == "Linux" and pytorch_device == "cuda":
                hw_info.has_rocm = True
                hw_info.compute_device = "cuda"  # ROCm uses 'cuda' string
            else:
                hw_info.compute_device = "cpu"  # Windows fallback

    # 4. Intel iGPU Detection
    intel_info = detect_intel_igpu_windows()
    if intel_info:
        hw_info.has_intel_igpu = True
        if not hw_info.gpu_name:
            hw_info.gpu_name = intel_info["name"]
        # VRAM/Shared Memory uebernehmen falls noch nicht gesetzt
        if not hw_info.gpu_memory_gb and intel_info.get("vram_gb"):
            hw_info.gpu_memory_gb = intel_info["vram_gb"]

    # 5. ONNX Runtime Provider Detection
    hw_info.onnx_providers = detect_onnx_providers()

    # Check for DirectML
    if "DmlExecutionProvider" in hw_info.onnx_providers:
        hw_info.has_directml = True
    device_str = str(hw_info.pytorch_device or "")
    if device_str.startswith("dml") or "privateuseone" in device_str:
        hw_info.compute_device = "dml"
        hw_info.has_directml = True
        if not hw_info.gpu_name:
            hw_info.gpu_name = _get_directml_device_name()

    # 6. Calculate recommended settings
    recommended = get_recommended_settings(hw_info)
    hw_info.recommended_batch_size = recommended["batch_size"]
    hw_info.recommended_num_workers = recommended["num_workers"]

    # 7. Summary
    logger.info("=" * 60)
    logger.info("Hardware-Erkennung abgeschlossen:")
    logger.info(f"  Platform: {hw_info.platform}")
    logger.info(f"  CPU Cores: {hw_info.cpu_count}")
    logger.info(f"  PyTorch Device: {hw_info.pytorch_device}")

    if hw_info.gpu_name:
        logger.info(f"  GPU: {hw_info.gpu_name}")
        if hw_info.gpu_memory_gb:
            logger.info(f"  GPU Memory: {hw_info.gpu_memory_gb:.2f} GB")

    if hw_info.has_nvidia_cuda:
        logger.info("  [OK] NVIDIA CUDA verfuegbar - volle GPU-Beschleunigung")
    elif hw_info.has_rocm:
        logger.info("  [OK] AMD ROCm verfuegbar - volle GPU-Beschleunigung")
    elif hw_info.has_amd_gpu:
        logger.info("  [WARN] AMD GPU (Windows) - CPU fuer ML, AMF fuer Video")
        if hw_info.has_directml:
            logger.info("  [OK] DirectML verfuegbar - AI-Beschleunigung via ONNX")
    elif hw_info.has_intel_igpu:
        logger.info("  [WARN] Intel iGPU - CPU fuer ML, QSV fuer Video")
        if hw_info.has_directml:
            logger.info("  [OK] DirectML verfuegbar - AI-Beschleunigung via ONNX")
    else:
        logger.info("  [WARN] Keine dedizierte GPU - CPU-Modus")

    if hw_info.onnx_providers and len(hw_info.onnx_providers) > 0:
        top_providers = hw_info.onnx_providers[:2]
        logger.info(f"  ONNX Providers: {', '.join(top_providers)}")

    logger.info(f"  Recommended Batch Size: {hw_info.recommended_batch_size}")
    logger.info(f"  Recommended Workers: {hw_info.recommended_num_workers}")
    logger.info("=" * 60)

    return hw_info


def detect_directml_device() -> str | None:
    """
    Erkennt ein verfuegbares DirectML-Geraet.
    """
    if platform.system() != "Windows":
        return None
    try:
        import torch_directml

        if torch_directml.is_available():
            device = torch_directml.device()
            logger.info(f"DirectML-Geraet gefunden: {torch_directml.device_name(0)}")
            return str(device)
    except ImportError:
        logger.debug("torch_directml ist nicht installiert.")
    except OSError as e:
        # DLL load failure
        logger.warning(f"DirectML DLL-Fehler: {e}")
    except Exception as e:
        logger.error(f"Fehler bei der DirectML-Geraeteerkennung: {e}")
    return None


def _get_directml_device_name() -> str | None:
    """Gibt den DirectML-Geraetenamen zurueck (falls verfuegbar)."""
    if platform.system() != "Windows":
        return None
    try:
        import torch_directml

        if torch_directml.is_available():
            return torch_directml.device_name(0)
    except Exception:
        return None
    return None


# Globale Hardware-Info (wird bei erstem Aufruf initialisiert)
_hardware_info: HardwareInfo | None = None


def get_hardware_info(refresh: bool = False) -> HardwareInfo:
    """
    Gibt die Hardware-Informationen zurück.

    Args:
        refresh: Falls True, wird Hardware neu erkannt

    Returns:
        HardwareInfo-Objekt
    """
    global _hardware_info
    if _hardware_info is None or refresh:
        _hardware_info = detect_hardware()
    return _hardware_info


def get_device() -> str:
    """
    Ermittelt das beste verfügbare Compute-Device.
    Priorität: CUDA > DirectML > MPS > CPU
    """
    hw_info = get_hardware_info()
    device = hw_info.pytorch_device or "cpu"
    device_str = str(device)

    # Konvertiert "privateuseone:0" (DirectML) zu "dml"
    if "privateuseone" in device_str:
        return "dml"

    return device_str


def get_device_info() -> dict:
    """
    Get device information as dictionary for compatibility with AI modules.

    Returns:
        Dictionary with device information including device_type, gpu_info, etc.
    """
    hw_info = get_hardware_info()

    # Determine device type
    device_type = "cpu"  # Default
    if hw_info.has_nvidia_cuda:
        device_type = "cuda"
    elif hw_info.has_amd_gpu:
        device_type = "directml"
    elif hw_info.has_intel_igpu:
        device_type = "directml"  # Intel also uses DirectML on Windows

    # Create GPU info dictionaries for compatibility
    nvidia_gpu_info = None
    if hw_info.has_nvidia_cuda:
        nvidia_gpu_info = {
            "name": hw_info.gpu_name,
            "memory_gb": hw_info.gpu_memory_gb,
            "cuda_version": hw_info.cuda_version,
            "available": True,
        }

    amd_gpu_info = None
    if hw_info.has_amd_gpu:
        amd_gpu_info = {
            "name": hw_info.gpu_name,
            "memory_gb": hw_info.gpu_memory_gb,
            "rocm_available": hw_info.has_rocm,
            "directml_available": hw_info.has_directml,
            "available": True,
        }

    intel_gpu_info = None
    if hw_info.has_intel_igpu:
        intel_gpu_info = {
            "name": hw_info.gpu_name,
            "memory_gb": hw_info.gpu_memory_gb,
            "directml_available": hw_info.has_directml,
            "available": True,
        }

    return {
        "device_type": device_type,
        "pytorch_device": str(hw_info.pytorch_device) if hw_info.pytorch_device else "cpu",
        "has_gpu": hw_info.has_nvidia_cuda or hw_info.has_amd_gpu or hw_info.has_intel_igpu,
        "nvidia_gpu_info": nvidia_gpu_info,
        "amd_gpu_info": amd_gpu_info,
        "intel_gpu_info": intel_gpu_info,
        "cpu_info": {"cores": hw_info.cpu_count, "architecture": platform.machine() or "unknown"},
        "memory_gb": hw_info.gpu_memory_gb,  # Use GPU memory, could add system memory later
        "os_name": hw_info.platform,
        "onnx_providers": hw_info.onnx_providers,
    }
