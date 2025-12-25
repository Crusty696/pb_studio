"""
PB Studio Hardware Bootstrapper
-------------------------------
Responsible for initial hardware detection and environment configuration before the main app loads.
Determines if the system should run on CUDA (NVIDIA) or DirectML (AMD/Intel).
"""

import importlib.util
import logging
import os
import platform
import sys
from typing import Literal

# Configure basic logging for boot phase
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Bootstrapper")

HardwareStrategy = Literal["cuda", "directml", "cpu"]

class Bootstrapper:
    def __init__(self):
        self.system = platform.system()
        self.strategy: HardwareStrategy = "cpu"
        self.env_vars: dict[str, str] = {}

    def check_requirements(self) -> bool:
        """Check critical validation requirements."""
        # 1. Check Python Version (implicitly handled by dependencies, but good for sanity)
        
        return True

    def detect_hardware(self) -> HardwareStrategy:
        """
        Detects available hardware and determines the best execution strategy.
        Returns: 'cuda', 'directml', or 'cpu'
        """
        logger.info(f"System: {self.system}")

        # Check for NVIDIA / CUDA
        if self._is_cuda_available():
            logger.info("Hardware detected: NVIDIA GPU (CUDA)")
            return "cuda"

        # Check for DirectML (AMD / Intel on Windows)
        if self.system == "Windows" and self._is_directml_available():
            logger.info("Hardware detected: AMD/Intel GPU (DirectML)")
            return "directml"

        logger.warning("No GPU acceleration found. Falling back to CPU (Slow).")
        return "cpu"

    def _is_cuda_available(self) -> bool:
        """Check if torch.cuda is available."""
        try:
            import torch
            if torch.cuda.is_available():
                return True
        except ImportError:
            pass
        return False

    def _is_directml_available(self) -> bool:
        """
        Check if onnxruntime-directml is available and supported.
        Also validates onnxruntime-genai availability.
        """
        try:
            # 1. Base ONNX Runtime Check
            if importlib.util.find_spec("onnxruntime") is None:
                return False

            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'DmlExecutionProvider' not in providers:
                return False

            # 2. GenAI Check (for V2 features)
            # If we want to use Phi-3 via DirectML, this must import without error
            if importlib.util.find_spec("onnxruntime_genai") is not None:
                logger.info("DirectML GenAI runtime detected.")
            else:
                logger.warning("DirectML detected, but onnxruntime-genai is missing. Phi-3 might fail.")

            return True

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"DirectML check failed: {e}")
        return False

    def configure_environment(self, strategy: HardwareStrategy):
        """Sets environment variables based on strategy."""
        self.strategy = strategy

        # Set global flag for the app
        os.environ["PB_HARDWARE_STRATEGY"] = strategy

        if strategy == "cuda":
            os.environ["ORT_STRATEGY"] = "cuda"
            # Potential CUDA specific optimizations could go here
        elif strategy == "directml":
            os.environ["ORT_STRATEGY"] = "directml"
            # FIX: Force dedicated GPU, ignore integrated GPU (iGPU)
            self._configure_dedicated_gpu()
        else:
            os.environ["ORT_STRATEGY"] = "cpu"

        logger.info(f"Environment configured for: {strategy.upper()}")

    def _configure_dedicated_gpu(self):
        """
        Configure DirectML to use ONLY the dedicated GPU (e.g., RX 7800 XT).
        Ignores integrated GPUs (iGPU like AMD Radeon Graphics on CPU).

        Sets DML_VISIBLE_DEVICES to the index of the dedicated GPU.
        """
        try:
            import torch_directml

            if not torch_directml.is_available():
                return

            device_count = torch_directml.device_count()
            if device_count <= 1:
                return  # Only one GPU, no need to filter

            dedicated_idx = None
            dedicated_name = None
            best_score = -1

            for i in range(device_count):
                name = torch_directml.device_name(i)
                name_lower = name.lower() if name else ""

                # Score: dedicated GPUs (RX, XT) get high score, iGPU gets negative
                score = 0

                # RX-Karten sind dediziert
                if " rx " in name_lower or name_lower.startswith("rx "):
                    score += 1000

                # XT-Suffix = höhere Leistungsklasse
                if " xt" in name_lower:
                    score += 100

                # VRAM-Schätzung aus Namen (7800 = 16GB, etc.)
                for model in ["7900", "7800", "6900", "6800"]:
                    if model in name_lower:
                        score += 500
                        break

                # Integrierte GPUs (nur "Graphics" ohne RX) STARK abwerten
                if "graphics" in name_lower and "rx" not in name_lower:
                    score -= 10000  # Effektiv ausschließen

                logger.debug(f"DirectML Device {i}: {name} (Score: {score})")

                if score > best_score:
                    best_score = score
                    dedicated_idx = i
                    dedicated_name = name

            if dedicated_idx is not None and best_score > 0:
                # Set environment variable to force this GPU
                os.environ["DML_VISIBLE_DEVICES"] = str(dedicated_idx)
                os.environ["PB_GPU_INDEX"] = str(dedicated_idx)
                os.environ["PB_GPU_NAME"] = dedicated_name or "Unknown"
                
                # CRITICAL: Set remapping flag. Since DML_VISIBLE_DEVICES is set,
                # the process now sees this device as Index 0.
                os.environ["PB_GPU_REMAPPED"] = "1"
                
                logger.info(f"✓ Dedicated GPU forced: {dedicated_name} (Index {dedicated_idx})")
                logger.info(f"  → Integrated GPU (iGPU) will be IGNORED (Logical Index will be 0)")
            else:
                logger.warning("No dedicated GPU found, using default DirectML device")

        except ImportError:
            logger.debug("torch_directml not available for GPU detection")
        except Exception as e:
            logger.warning(f"Dedicated GPU detection failed: {e}")

    def run(self) -> bool:
        """Main execution method."""
        logger.info("Starting PB Studio Bootstrapper...")

        if not self.check_requirements():
            return False

        strategy = self.detect_hardware()
        self.configure_environment(strategy)

        return True

if __name__ == "__main__":
    boot = Bootstrapper()
    boot.run()