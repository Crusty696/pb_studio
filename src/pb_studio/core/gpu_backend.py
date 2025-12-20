"""
GPU Backend Factory - Automatische GPU-Erkennung und Backend-Konfiguration.

Vereinfachte API fuer KI-Module (YOLO, CLIP, Demucs).

Usage:
    from pb_studio.core.gpu_backend import get_device, get_onnx_providers, get_backend_info

    device = get_device()           # cuda, privateuseone:0, cpu
    providers = get_onnx_providers() # [DmlExecutionProvider, CPUExecutionProvider]
    info = get_backend_info()       # Dict mit allen Infos
"""

import threading
from enum import Enum

import torch

from ..utils.logger import get_logger

logger = get_logger(__name__)

# MEDIUM-07 FIX: Thread-Lock fuer Singleton
_singleton_lock = threading.Lock()

# LOW-05: Konfigurierbare ONNX Provider Prioritaet
DEFAULT_ONNX_PROVIDER_PRIORITY = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
    "OpenVINOExecutionProvider",
    "CPUExecutionProvider",
]


class GPUVendor(Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    NONE = "none"


class GPUBackend(Enum):
    CUDA = "cuda"
    ROCM = "rocm"
    DIRECTML = "directml"
    MPS = "mps"
    CPU = "cpu"


class GPUBackendFactory:
    _instance = None

    def __new__(cls):
        # MEDIUM-07 FIX: Thread-safe Singleton
        if cls._instance is None:
            with _singleton_lock:
                # Double-check nach Lock-Acquisition
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._backend = GPUBackend.CPU
        self._vendor = GPUVendor.NONE
        self._device = "cpu"
        self._device_name = None
        self._onnx_providers = []
        self._detect_hardware()

    def _detect_hardware(self):
        logger.info("GPU Backend Detection gestartet...")
        if self._try_cuda():
            return
        if self._try_directml():
            return
        if self._try_mps():
            return
        logger.warning("Kein GPU-Backend verfuegbar, nutze CPU Fallback")
        self._setup_cpu()

    def _detect_rocm(self) -> bool:
        """Erkenne ROCm (AMD) Backend mit expliziten Pruefungen."""
        try:
            # 1. Explizite HIP-Pruefung (ROCm's CUDA-Kompatibilitaetsschicht)
            if hasattr(torch.version, "hip") and torch.version.hip:
                logger.info(f"ROCm detected: HIP version {torch.version.hip}")
                return True

            # 2. GPU-Name pruefen (bei torch.cuda.is_available())
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0).lower()
                if "amd" in device_name or "radeon" in device_name:
                    logger.info(f"AMD GPU detected via device name: {device_name}")
                    return True

            # 3. Fallback: Kein CUDA-Version aber cuda verfuegbar = wahrscheinlich ROCm
            if torch.cuda.is_available() and not torch.version.cuda:
                logger.info("ROCm detected: CUDA API available but no CUDA version string")
                return True

            return False
        except Exception as e:
            logger.debug(f"ROCm detection failed: {e}")
            return False

    def _try_cuda(self):
        """Versuche CUDA/ROCm Backend zu initialisieren."""
        try:
            if torch.cuda.is_available():
                self._device_name = torch.cuda.get_device_name(0)

                # Explizite ROCm-Detection ZUERST
                if self._detect_rocm():
                    self._backend = GPUBackend.ROCM
                    self._vendor = GPUVendor.AMD
                    self._device = "cuda"  # ROCm nutzt torch.cuda API
                    logger.info("=== ROCm Backend aktiviert ===")
                    logger.info(f"AMD GPU: {self._device_name}")
                else:
                    # NVIDIA CUDA
                    self._backend = GPUBackend.CUDA
                    self._vendor = GPUVendor.NVIDIA
                    self._device = "cuda"
                    cuda_version = torch.version.cuda if torch.version.cuda else "unknown"
                    logger.info("=== CUDA Backend aktiviert ===")
                    logger.info(f"NVIDIA GPU: {self._device_name} (CUDA {cuda_version})")

                self._detect_onnx_providers()
                return True
        except Exception as e:
            logger.debug(f"CUDA/ROCm nicht verfuegbar: {e}")
        return False

    def _try_directml(self):
        try:
            import torch_directml

            if torch_directml.is_available():
                self._backend = GPUBackend.DIRECTML
                self._device = torch_directml.device()
                device_name = torch_directml.device_name(0)
                self._device_name = device_name

                # MEDIUM-08 FIX: Case-insensitive Vendor Detection
                device_name_upper = device_name.upper() if device_name else ""
                if "AMD" in device_name_upper or "RADEON" in device_name_upper:
                    self._vendor = GPUVendor.AMD
                elif "INTEL" in device_name_upper or "ARC" in device_name_upper:
                    self._vendor = GPUVendor.INTEL
                elif (
                    "NVIDIA" in device_name_upper
                    or "GEFORCE" in device_name_upper
                    or "RTX" in device_name_upper
                ):
                    self._vendor = GPUVendor.NVIDIA

                logger.info("=== DirectML Backend aktiviert ===")
                logger.info(f"GPU: {device_name} (Vendor: {self._vendor.value})")
                self._detect_onnx_providers()
                return True
        except ImportError:
            logger.debug("DirectML nicht installiert (torch_directml)")
        except Exception as e:
            logger.debug(f"DirectML nicht verfuegbar: {e}")
        return False

    def _try_mps(self):
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._backend = GPUBackend.MPS
                self._vendor = GPUVendor.APPLE
                self._device = "mps"
                self._device_name = "Apple Silicon"
                logger.info("=== MPS Backend aktiviert ===")
                logger.info("Apple Silicon GPU erkannt")
                self._detect_onnx_providers()
                return True
        except Exception as e:
            logger.debug(f"Apple MPS nicht verfuegbar: {e}")
        return False

    def _setup_cpu(self):
        import os

        self._backend = GPUBackend.CPU
        self._vendor = GPUVendor.NONE
        self._device = "cpu"
        num_threads = os.cpu_count() or 4
        self._device_name = f"CPU ({num_threads} cores)"
        torch.set_num_threads(num_threads)
        logger.info("=== CPU Backend aktiviert ===")
        logger.info(f"CPU: {num_threads} Threads konfiguriert")
        self._detect_onnx_providers()

    def _detect_onnx_providers(self, priority: list[str] = None):
        """
        Erkennt verfuegbare ONNX Runtime Provider.

        Args:
            priority: Optionale benutzerdefinierte Prioritaetsliste
        """
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
            logger.debug(f"Alle verfuegbaren ONNX Providers: {available}")

            # LOW-05 FIX: Nutze konfigurierbare Prioritaet
            provider_priority = priority or DEFAULT_ONNX_PROVIDER_PRIORITY

            # Detailliertes Logging der Provider-Auswahl
            selected = []
            for provider in provider_priority:
                if provider in available:
                    selected.append(provider)
                    logger.debug(f"Provider '{provider}' verfuegbar und ausgewaehlt")
                else:
                    logger.debug(f"Provider '{provider}' nicht verfuegbar")

            self._onnx_providers = selected

            # Fallback-Logging
            if not self._onnx_providers:
                self._onnx_providers = ["CPUExecutionProvider"]
                logger.warning("Kein bevorzugter Provider verfuegbar, nutze CPU")

            logger.info(f"Ausgewaehlte ONNX Providers: {self._onnx_providers}")
        except ImportError:
            self._onnx_providers = []
            logger.warning("ONNX Runtime nicht installiert")

    @property
    def backend(self):
        return self._backend

    @property
    def vendor(self):
        return self._vendor

    @property
    def device(self):
        return self._device

    @property
    def device_name(self):
        return self._device_name or "Unknown"

    @property
    def onnx_providers(self):
        return self._onnx_providers.copy()

    @property
    def is_gpu_available(self):
        return self._backend != GPUBackend.CPU

    def get_info(self):
        return {
            "backend": self._backend.value,
            "vendor": self._vendor.value,
            "device": str(self._device),
            "device_name": self._device_name,
            "onnx_providers": self._onnx_providers,
            "is_gpu": self.is_gpu_available,
        }


_gpu_backend = None


def _get_factory():
    global _gpu_backend
    if _gpu_backend is None:
        _gpu_backend = GPUBackendFactory()
    return _gpu_backend


def get_device():
    return _get_factory().device


def get_onnx_providers():
    return _get_factory().onnx_providers


def get_backend():
    return _get_factory().backend.value


def get_vendor():
    return _get_factory().vendor.value


def get_backend_info():
    return _get_factory().get_info()


def is_gpu_available():
    return _get_factory().is_gpu_available


def print_backend_info():
    """Gibt Backend-Infos auf der Konsole aus."""
    info = get_backend_info()
    print("\n" + "=" * 50)
    print("  PB Studio - GPU Backend Info")
    print("=" * 50)
    print(f"  Backend:     {info['backend']}")
    print(f"  Vendor:      {info['vendor']}")
    print(f"  Device:      {info['device']}")
    print(f"  Device Name: {info['device_name']}")
    print(f"  GPU aktiv:   {'Ja' if info['is_gpu'] else 'Nein'}")
    if info["onnx_providers"]:
        print(f"  ONNX:        {', '.join(info['onnx_providers'][:2])}")
    print("=" * 50 + "\n")
