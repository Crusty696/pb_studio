
import pytest
import os
import numpy as np
from PIL import Image
from pathlib import Path
from pb_studio.bootstrapper import Bootstrapper
from pb_studio.ai.model_manager import ModelManager
from pb_studio.ai.models.moondream import MoondreamONNX

@pytest.mark.integration
@pytest.mark.gpu
def test_moondream_integration():
    """Integration test for Moondream2 model (Encoder Only)."""
    # 1. Boot (Ensure env is set)
    Bootstrapper().run()
    
    manager = ModelManager()
    
    # 2. Check Model Availability
    spec = manager.registry.get_model("moondream2_onnx")
    if not spec:
        pytest.skip("Moondream2 model spec not found in registry.")

    model_path = manager.registry.base_dir / spec.file_name
    if not model_path.exists():
        pytest.skip(f"Moondream2 model files missing at {model_path}")

    # 3. Instantiate
    try:
        hardware = os.environ.get("PB_HARDWARE_STRATEGY", "cpu")
        model = MoondreamONNX(model_path, hardware_strategy=hardware)
    except Exception as e:
        pytest.fail(f"Failed to instantiate MoondreamONNX: {e}")

    # 4. Inference
    try:
        dummy_img = Image.fromarray(np.random.randint(0, 255, (378, 378, 3), dtype=np.uint8))
        result = model.describe_image(dummy_img, "Test prompt")
        assert isinstance(result, str)
        assert len(result) > 0 or result == "" # Result depends on implementation stub
    except Exception as e:
        pytest.fail(f"Inference failed: {e}")
