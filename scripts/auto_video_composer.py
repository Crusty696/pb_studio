import os
import subprocess
import random
import glob
from pathlib import Path

# --- Configuration ---
AUDIO_DIR = r"C:\Users\david\Videos\Music-Video_Clips\AV\Audio"
VIDEO_DIR = r"C:\Users\david\Videos\Music-Video_Clips\AV\Video"
OUTPUT_DIR = r"C:\Users\david\Videos\Music-Video_Clips\AV\output"
FFMPEG_CMD = "ffmpeg" 
TARGET_W = 1920
TARGET_H = 1080
TARGET_FPS = 30
VIDEO_BITRATE = "5000k"
AUDIO_BITRATE = "192k"
LOG_FILE = os.path.join(OUTPUT_DIR, "render_log.txt")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log_message(message):
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def get_video_duration(video_path):
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except:
        return 0

def create_video_for_audio(audio_path):
    filename = os.path.basename(audio_path)
    output_path = os.path.join(OUTPUT_DIR, f"Final_1h_Plus_{filename}.mp4")
    
    log_message(f"--- Processing Full Audio (1h+ Target): {filename} ---")

    audio_duration = get_video_duration(audio_path)
    if audio_duration == 0:
        log_message("  [ERROR] Failed to get audio duration.")
        return
    
    log_message(f"  Total Duration: {audio_duration:.2f}s ({audio_duration/3600:.2f} hours)")

    video_files = glob.glob(os.path.join(VIDEO_DIR, "**/*.mp4"), recursive=True)
    if not video_files:
        log_message("  [ERROR] No video files found!")
        return

    selected_clips = []
    current_duration = 0
    # Use a fixed seed for reproducibility if needed, or just shuffle
    random.shuffle(video_files)

    log_message("  Selecting clips to fill the entire duration...")
    while current_duration < audio_duration:
        for vid in video_files:
            dur = get_video_duration(vid)
            if dur > 0:
                selected_clips.append(vid)
                current_duration += dur
            if current_duration >= audio_duration:
                break
        if current_duration == 0:
            break
    
    log_message(f"  Selected {len(selected_clips)} clips. Total video capacity: {current_duration:.2f}s")

    concat_list_path = os.path.join(OUTPUT_DIR, f"concat_full.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for vid in selected_clips:
            safe_path = vid.replace(os.sep, "/")
            f.write(f"file '{safe_path}'\n")

    vf = (
        f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2,"
        f"fps={TARGET_FPS},format=yuv420p"
    )

    cmd = [
        FFMPEG_CMD, "-y",
        "-f", "concat", "-safe", "0", "-i", concat_list_path,
        "-i", audio_path,
        "-map", "0:v", "-map", "1:a",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-b:v", VIDEO_BITRATE,
        "-c:a", "aac", "-b:a", AUDIO_BITRATE,
        "-shortest", output_path
    ]

    log_message(f"  STARTING FULL RENDER to {output_path}...")
    log_message("  Note: This process is running in the background. Check logs for progress.")
    
    try:
        # We run this and redirect output to log file so we don't hang
        with open(LOG_FILE, "a") as log_f:
            subprocess.Popen(cmd, stdout=log_f, stderr=log_f)
        log_message("  [INFO] Render process started in background.")
    except Exception as e:
        log_message(f"  [ERROR] Failed to start process: {e}")

def main():
    audio_files = []
    for root, dirs, files in os.walk(AUDIO_DIR):
        for file in files:
            if file.lower().endswith((".wav", ".mp3", ".m4a", ".flac")):
                audio_files.append(os.path.join(root, file))
    
    if audio_files:
        # Choose the largest file to ensure 1h+
        audio_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
        create_video_for_audio(audio_files[0])

if __name__ == "__main__":
    main()