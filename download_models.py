
import logging
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from pb_studio.utils.logger import setup_logging
from pb_studio.bootstrapper import Bootstrapper
from pb_studio.ai.model_manager import ModelManager

# Setup basic logging
setup_logging(console_level=logging.INFO)
logger = logging.getLogger("ModelDownloader")

def download_models():
    # 1. Bootstrapper (to set env vars correctly)
    boot = Bootstrapper()
    boot.run()
    
    manager = ModelManager()
    
    # 2. Download Moondream
    logger.info(">>> Starting Moondream2 Download...")
    hw_models = manager.get_hardware_optimized_models() # Initialize/Check
    
    path_md = manager.download_model("moondream2_onnx")
    if path_md:
        logger.info(f"✅ Moondream2 ready at: {path_md}")
    else:
        logger.error("❌ Moondream2 download failed.")

    # 3. Download Phi-3
    logger.info(">>> Starting Phi-3 Mini Download (This may take a while)...")
    path_phi = manager.download_model("phi-3-mini-4k-instruct-onnx")
    if path_phi:
        logger.info(f"✅ Phi-3 Mini ready at: {path_phi}")
    else:
        logger.error("❌ Phi-3 Mini download failed.")

if __name__ == "__main__":
    download_models()
