
import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add src to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from pb_studio.bootstrapper import Bootstrapper, HardwareStrategy

class TestBootstrapper(unittest.TestCase):

    def setUp(self):
        """Set up a fresh Bootstrapper instance for each test."""
        self.bootstrapper = Bootstrapper()
        # Clear any environment variables that might be set
        os.environ.pop("PB_HARDWARE_STRATEGY", None)
        os.environ.pop("ORT_STRATEGY", None)
        os.environ.pop("DML_VISIBLE_DEVICES", None)
        os.environ.pop("PB_GPU_INDEX", None)
        os.environ.pop("PB_GPU_NAME", None)
        os.environ.pop("PB_GPU_REMAPPED", None)


    @patch('pb_studio.bootstrapper.Bootstrapper._is_cuda_available', return_value=True)
    @patch('pb_studio.bootstrapper.Bootstrapper._is_directml_available', return_value=False)
    def test_detect_hardware_cuda(self, mock_directml, mock_cuda):
        """Test hardware detection for CUDA."""
        strategy = self.bootstrapper.detect_hardware()
        self.assertEqual(strategy, "cuda")

    @patch('pb_studio.bootstrapper.Bootstrapper._is_cuda_available', return_value=False)
    @patch('pb_studio.bootstrapper.Bootstrapper._is_directml_available', return_value=True)
    def test_detect_hardware_directml(self, mock_directml, mock_cuda):
        """Test hardware detection for DirectML."""
        self.bootstrapper.system = "Windows"
        strategy = self.bootstrapper.detect_hardware()
        self.assertEqual(strategy, "directml")

    @patch('pb_studio.bootstrapper.Bootstrapper._is_cuda_available', return_value=False)
    @patch('pb_studio.bootstrapper.Bootstrapper._is_directml_available', return_value=False)
    def test_detect_hardware_cpu(self, mock_directml, mock_cuda):
        """Test hardware detection for CPU fallback."""
        strategy = self.bootstrapper.detect_hardware()
        self.assertEqual(strategy, "cpu")

    def test_configure_environment_cuda(self):
        """Test environment configuration for CUDA."""
        self.bootstrapper.configure_environment("cuda")
        self.assertEqual(os.environ["PB_HARDWARE_STRATEGY"], "cuda")
        self.assertEqual(os.environ["ORT_STRATEGY"], "cuda")

    def test_configure_environment_directml(self):
        """Test environment configuration for DirectML."""
        with patch('pb_studio.bootstrapper.Bootstrapper._configure_dedicated_gpu') as mock_configure_gpu:
            self.bootstrapper.configure_environment("directml")
            self.assertEqual(os.environ["PB_HARDWARE_STRATEGY"], "directml")
            self.assertEqual(os.environ["ORT_STRATEGY"], "directml")
            mock_configure_gpu.assert_called_once()

    def test_configure_environment_cpu(self):
        """Test environment configuration for CPU."""
        self.bootstrapper.configure_environment("cpu")
        self.assertEqual(os.environ["PB_HARDWARE_STRATEGY"], "cpu")
        self.assertEqual(os.environ["ORT_STRATEGY"], "cpu")

    @patch('importlib.util.find_spec')
    def test_is_directml_available_success(self, mock_find_spec):
        """Test _is_directml_available success case."""
        mock_find_spec.return_value = True
        with patch('onnxruntime.get_available_providers', return_value=['DmlExecutionProvider']):
            self.assertTrue(self.bootstrapper._is_directml_available())

    @patch('importlib.util.find_spec', return_value=None)
    def test_is_directml_available_no_onnxruntime(self, mock_find_spec):
        """Test _is_directml_available when onnxruntime is not installed."""
        self.assertFalse(self.bootstrapper._is_directml_available())
        
    @patch('torch_directml.is_available', return_value=True)
    @patch('torch_directml.device_count', return_value=2)
    @patch('torch_directml.device_name')
    def test_configure_dedicated_gpu_selection(self, mock_device_name, mock_device_count, mock_is_available):
        """Test dedicated GPU selection logic."""
        # Mock two GPUs: one integrated, one dedicated
        mock_device_name.side_effect = ["AMD Radeon Graphics", "AMD Radeon RX 7800 XT"]
        
        self.bootstrapper._configure_dedicated_gpu()

        # Check that the dedicated GPU is selected and environment variables are set
        self.assertEqual(os.environ.get("DML_VISIBLE_DEVICES"), "1")
        self.assertEqual(os.environ.get("PB_GPU_INDEX"), "1")
        self.assertEqual(os.environ.get("PB_GPU_NAME"), "AMD Radeon RX 7800 XT")
        self.assertEqual(os.environ.get("PB_GPU_REMAPPED"), "1")

if __name__ == '__main__':
    unittest.main()
