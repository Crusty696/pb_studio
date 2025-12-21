
import sys
import pytest
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

def test_imports():
    """Smoke test: Verify critical modules can be imported."""
    try:
        import pb_studio.bootstrapper
        import pb_studio.ai.model_manager
        import pb_studio.database.connection
        import pb_studio.gui.main_window
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_project_structure():
    """Verify critical directories exist."""
    root = Path.cwd()
    assert (root / "src").exists()
    assert (root / "data").exists()
    assert (root / "pyproject.toml").exists()
