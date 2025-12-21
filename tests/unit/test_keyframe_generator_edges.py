
import unittest
from pb_studio.pacing.keyframe_generator import KeyframeGenerator

class TestKeyframeGeneratorEdges(unittest.TestCase):
    def test_zoom_empty_beats(self):
        """Test zoom generation with empty beats list."""
        result = KeyframeGenerator.generate_zoom_curve([], 1.0, 30)
        self.assertEqual(result, "0: (1.0)")

    def test_shake_empty_beats(self):
        """Test shake generation with empty beats list."""
        result = KeyframeGenerator.generate_shake_curve([], 1.0, 30)
        self.assertEqual(result, "0: (0)")

    def test_zoom_zero_intensity(self):
        """Test zoom with 0 intensity (should be flat)."""
        beats = [1.0]
        result = KeyframeGenerator.generate_zoom_curve(beats, 0.0, 30)
        # 30: (1.0 + 0.05 * 0) => 1.0
        self.assertIn("30: (1.0)", result)

    def test_high_fps(self):
        """Test with high FPS."""
        beats = [1.0]
        # 1.0s * 60fps = frame 60
        result = KeyframeGenerator.generate_zoom_curve(beats, 1.0, 60)
        self.assertIn("60: (1.05)", result)

if __name__ == "__main__":
    unittest.main()
