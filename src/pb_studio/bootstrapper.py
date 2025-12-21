
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
        else:
            os.environ["ORT_STRATEGY"] = "cpu"

        logger.info(f"Environment configured for: {strategy.upper()}")

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
