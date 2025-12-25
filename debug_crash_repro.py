import os
import sys
import logging
import faulthandler
from pathlib import Path

# Enable faulthandler
faulthandler.enable()

# Add src to path
project_root = Path(r"c:\GEMINI_PROJEKTE\_Pb-studio_V_2\pb_studio")
sys.path.insert(0, str(project_root / "src"))

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Import generator
try:
    from pb_studio.video.thumbnail_generator import ThumbnailGenerator
    from pb_studio.utils.video_utils import get_video_info_safe
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

# Target file
target_file = r"C:\Users\david\Videos\Music-Video_Clips\AV\Video\generation 4\20250614_0320_Enchanted_Cavern_Dance_gen_01jxp78ab9fwca6vc381kzkn9v.mp4"

print(f"Testing crash on file: {target_file}")

if not os.path.exists(target_file):
    print("File not found!")
    sys.exit(1)

gen = ThumbnailGenerator(cache_dir="crash_test_thumbs")

print("Attempting to reproduce crash (Double Check HW Decode)...")

try:
    # Run once with DEBUG to see logs
    print(f"Checking info for: {target_file}")
    
    # 1. Get Info
    info = get_video_info_safe(target_file)
    if info:
        print(f"Info: {info}")
    else:
        print("Failed to get info")
        
    # 2. Generate Thumb
    res = gen.generate(target_file, force_regenerate=True)
    print(f"Thumb path: {res}")
            
except Exception as e:
    print(f"Caught exception: {e}")

print("Debug run done.")
