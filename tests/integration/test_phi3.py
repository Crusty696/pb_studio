
import pytest
import os
from pathlib import Path
from pb_studio.bootstrapper import Bootstrapper
from pb_studio.ai.model_manager import ModelManager
from pb_studio.ai.models.phi3 import Phi3ONNX

@pytest.mark.integration
@pytest.mark.gpu
def test_phi3_integration():
    """Integration test for Phi-3 GenAI model."""
    # 1. Boot
    Bootstrapper().run()
    
    manager = ModelManager()
    
    # 2. Check Registry
    spec = manager.registry.get_model("phi-3-mini-4k-instruct-onnx")
    if not spec:
        pytest.skip("Phi-3 model spec not found.")

    model_path = manager.registry.base_dir / spec.file_name
    if not model_path.exists():
        pytest.skip(f"Phi-3 model files missing at {model_path}")

    # 3. Instantiate
    try:
        hardware = os.environ.get("PB_HARDWARE_STRATEGY", "cpu")
        # Ensure we don't crash if GenAI dlls are missing, but skip test
        try:
            import onnxruntime_genai
        except ImportError:
            pytest.skip("onnxruntime_genai not installed.")

        model = Phi3ONNX(model_path, hardware_strategy=hardware)
    except ImportError:
         pytest.skip("onnxruntime-genai missing")
    except Exception as e:
        pytest.fail(f"Failed to instantiate Phi3ONNX: {e}")

    # 4. Inference
    try:
        prompts = ["Test prompt for story."]
        result = model.generate_story(prompts)
        assert isinstance(result, str)
        assert len(result) > 0
    except Exception as e:
        pytest.fail(f"Inference failed: {e}")
