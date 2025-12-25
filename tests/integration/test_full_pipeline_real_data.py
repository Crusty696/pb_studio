import os
import sys
import logging
import shutil
from pathlib import Path

# Add src to python path so we can import pb_studio modules
sys.path.append(os.path.abspath("src"))

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("real_pipeline_test.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PipelineTest")

# --- Configuration ---
# USING REAL USER DATA
REAL_AUDIO_DIR = r"C:\Users\david\Videos\Music-Video_Clips\AV\Audio"
REAL_VIDEO_DIR = r"C:\Users\david\Videos\Music-Video_Clips\AV\Video"
TEST_OUTPUT_DIR = r"C:\Users\david\Videos\Music-Video_Clips\AV\output\PipelineTest"

# Duration for the test (process only first 5 minutes to prove it works)
TEST_DURATION_SEC = 300.0 

def find_first_audio():
    for root, dirs, files in os.walk(REAL_AUDIO_DIR):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3')):
                return os.path.join(root, file)
    return None

def find_videos():
    videos = []
    for root, dirs, files in os.walk(REAL_VIDEO_DIR):
        for file in files:
            if file.lower().endswith('.mp4'):
                videos.append(os.path.join(root, file))
    return videos

def run_pipeline():
    logger.info("="*50)
    logger.info("STARTING FULL PB STUDIO PIPELINE TEST (REAL DATA)")
    logger.info("="*50)

    # 0. Setup
    audio_path = find_first_audio()
    if not audio_path:
        logger.error("No audio found!")
        return
    
    videos = find_videos()
    if not videos:
        logger.error("No videos found!")
        return

    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Audio Source: {audio_path}")
    logger.info(f"Video Source: {len(videos)} clips found")
    logger.info(f"Output Dir:   {TEST_OUTPUT_DIR}")

    # ---------------------------------------------------------
    # 1. STEM SEPARATION (src.pb_studio.audio.stem_separator)
    # ---------------------------------------------------------
    logger.info("\n--- STEP 1: STEM SEPARATION ---")
    try:
        from pb_studio.audio.stem_separator import StemSeparator
        
        # Initialize with 'kuielab' (fast ONNX)
        separator = StemSeparator(model_preset="kuielab", cache_dir=Path(TEST_OUTPUT_DIR) / "stems")
        
        # Run separation (will trigger the AMD fix logic we implemented)
        # We only need stems for the first 5 minutes? 
        # StemSeparator typically processes full file, but we rely on caching.
        # This is the heavy part.
        
        logger.info("Starting Stem Separation... (This tests the AMD/DirectML Fix)")
        stems = separator.separate(audio_path)
        
        for name, path in stems.items():
            logger.info(f"  [OK] Generated Stem: {name} -> {path}")
            
    except Exception as e:
        logger.error(f"STEP 1 FAILED: {e}")
        # Continue? If stems fail, pacing might fallback, so we proceed to test resilience.

    # ---------------------------------------------------------
    # 2. AUDIO TRIGGER ANALYSIS (src.pb_studio.pacing.trigger_system)
    # ---------------------------------------------------------
    logger.info("\n--- STEP 2: TRIGGER ANALYSIS ---")
    try:
        from pb_studio.pacing.trigger_system import TriggerSystem
        
        # Initialize TriggerSystem (uses Stems if available)
        trigger_sys = TriggerSystem(use_stems=True, stem_cache_dir=Path(TEST_OUTPUT_DIR) / "stems")
        
        # Analyze first 5 minutes
        logger.info(f"Analyzing Triggers (0-{TEST_DURATION_SEC}s)...")
        triggers = trigger_sys.analyze_triggers(
            audio_path, 
            start_time=0.0, 
            end_time=TEST_DURATION_SEC
        )
        
        logger.info(f"  [OK] Detected {len(triggers.beat_times)} Beats")
        logger.info(f"  [OK] Detected {len(triggers.kick_times)} Kicks (from Stems)")
        logger.info(f"  [OK] BPM: {triggers.bpm}")

    except Exception as e:
        logger.error(f"STEP 2 FAILED: {e}")
        return

    # ---------------------------------------------------------
    # 3. AI PACING & SCENE DETECTION (src.pb_studio.pacing...)
    # ---------------------------------------------------------
    logger.info("\n--- STEP 3: PACING & SCENE DETECTION ---")
    try:
        from pb_studio.pacing.ai_enhanced_pacing_engine import AIEnhancedPacingEngine, AIPacingAnalysisResult
        from pb_studio.pacing.pacing_models import PacingCut
        
        engine = AIEnhancedPacingEngine()
        
        # Simulate Analysis Result object (normally created by full analysis)
        # We feed it our real trigger data
        analysis_result = AIPacingAnalysisResult(
            audio_features={"bpm": triggers.bpm},
            video_features={"tags": ["test"]},
            cross_modal_score=0.8,
            content_quality=0.9,
            recommended_strategy="dynamic",
            cut_suggestions=["kick", "beat"],
            pacing_confidence=0.9,
            dominant_scenes=[],
            scene_transitions=[],
            mood_consistency="high",
            ai_cut_points=triggers.kick_times[:20], # Use kicks as AI points
            quality_zones=[],
            energy_correlation=0.8
        )
        
        # Generate Cuts
        logger.info("Generating Cuts...")
        # Note: In a real run, we'd pass a specific video, but here we generate the abstract cut list first
        cuts = engine._generate_base_cuts(audio_path, TEST_DURATION_SEC, analysis_result)
        
        logger.info(f"  [OK] Generated {len(cuts)} Cuts based on Audio Triggers")
        
        # Assign Real Video Clips to Cuts (Simple Mapper for Test)
        # In real app: ClipMatcher does this.
        from pb_studio.pacing.pacing_models import CutListEntry
        
        final_cut_list = []
        import random
        
        for i, cut in enumerate(cuts):
            if i >= len(cuts) - 1: break
            
            duration = cuts[i+1].timestamp - cut.timestamp
            if duration < 0.5: continue # Skip too short
            
            video = random.choice(videos)
            
            entry = CutListEntry(
                clip_path=video,
                start_time=cut.timestamp,
                duration=duration,
                source_start=0.0, # Start from beginning of clip
                speed=1.0
            )
            final_cut_list.append(entry)
            
        logger.info(f"  [OK] Assigned {len(final_cut_list)} Video Clips to Cuts")

    except Exception as e:
        logger.error(f"STEP 3 FAILED: {e}", exc_info=True)
        return

    # ---------------------------------------------------------
    # 4. KEYFRAME GENERATION (src.pb_studio.pacing.keyframe_generator)
    # ---------------------------------------------------------
    logger.info("\n--- STEP 4: KEYFRAME STRING GENERATION ---")
    try:
        from pb_studio.pacing.keyframe_generator import KeyframeGenerator
        
        zoom_string = KeyframeGenerator.generate_zoom_curve(triggers.beat_times, intensity=1.5, fps=30)
        shake_string = KeyframeGenerator.generate_shake_curve(triggers.beat_times, intensity=1.0, fps=30)
        
        # Write to file
        kf_path = Path(TEST_OUTPUT_DIR) / "keyframes.txt"
        with open(kf_path, "w") as f:
            f.write(f"ZOOM:\n{zoom_string[:200]}...\n\n")
            f.write(f"SHAKE:\n{shake_string[:200]}...")
            
        logger.info(f"  [OK] Keyframe Strings generated and saved to {kf_path}")

    except Exception as e:
        logger.error(f"STEP 4 FAILED: {e}")

    # ---------------------------------------------------------
    # 5. RENDERING (src.pb_studio.video.video_renderer)
    # ---------------------------------------------------------
    logger.info("\n--- STEP 5: FINAL RENDERING ---")
    try:
        from pb_studio.video.video_renderer import VideoRenderer, RenderSettings
        
        output_video = Path(TEST_OUTPUT_DIR) / "Pipeline_Test_Result.mp4"
        
        # Use GPU if available
        settings = RenderSettings(use_gpu=True, preset="fast")
        renderer = VideoRenderer(settings)
        
        logger.info(f"Rendering to {output_video}...")
        
        success = renderer.render_video(
            cut_list=final_cut_list,
            audio_path=audio_path,
            output_path=str(output_video),
            progress_callback=lambda p: print(f"Render Progress: {int(p*100)}%", end="\r")
        )
        
        if success:
            logger.info(f"\n  [SUCCESS] Video Rendered Successfully!")
        else:
            logger.error(f"\n  [FAILED] Video Rendering returned False")

    except Exception as e:
        logger.error(f"STEP 5 FAILED: {e}", exc_info=True)

if __name__ == "__main__":
    run_pipeline()
