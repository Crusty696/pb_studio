
import pytest
from pb_studio.pacing.keyframe_generator import KeyframeGenerator

@pytest.mark.integration
def test_keyframe_generator_output_format():
    """Verify that KeyframeGenerator produces correct string formats for Deforum."""
    # Test Zoom
    beats = [1.0, 2.0, 3.0]
    result = KeyframeGenerator.generate_zoom_curve(beats, intensity=1.0, fps=30)
    
    # Expected pattern check
    assert "0: (1.0)" in result
    assert "30: (1.05)" in result
    assert "35: (1.0)" in result
    
    # Test Shake
    beats_shake = [1.0]
    result_shake = KeyframeGenerator.generate_shake_curve(beats_shake, intensity=2.0, fps=30)
    # 30: (20.0), 32: (-20.0), 34: (0)
    assert "30: (20.0)" in result_shake
    assert "32: (-20.0)" in result_shake
