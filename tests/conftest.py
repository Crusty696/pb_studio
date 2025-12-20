import sys
from pathlib import Path

# Add src to the Python path so that pb_studio packages can be imported in tests
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
