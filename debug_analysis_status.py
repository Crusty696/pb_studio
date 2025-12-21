
import sys
import os
from pathlib import Path
import json

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from pb_studio.database.connection import get_db_session
from pb_studio.database.models import VideoClip, Project
from pb_studio.database.models_analysis import ClipAnalysisStatus, ClipColors
from pb_studio.analysis.video_analyzer import VideoAnalyzer

def debug_analysis():
    print("--- Starting Debug Analysis ---")
    
    # 1. Setup DB
    session = get_db_session()
    
    # Find a test clip or create one
    clip = session.query(VideoClip).first()
    if not clip:
        print("No clips found in DB to test with.")
        return

    print(f"Testing with Clip ID: {clip.id}, Name: {clip.name}")
    print(f"Initial needs_reanalysis: {clip.needs_reanalysis}")
    
    # Check initial status
    status = session.query(ClipAnalysisStatus).filter(ClipAnalysisStatus.clip_id == clip.id).first()
    if status:
        print(f"Initial Analysis Status: Colors={status.colors_analyzed}, Scene={status.scene_analyzed}, Full={status.is_fully_analyzed()}")
    else:
        print("No initial analysis status record found.")

    # 2. Simulate Logic from ClipLibraryWidget._load_clips_from_db
    # This logic determines "unanalyzed_count"
    
    # Re-fetch with eager loading simulation (manually checking relationships)
    clip_db = session.query(VideoClip).filter(VideoClip.id == clip.id).first()
    
    analysis_data = {}
    is_analyzed = False
    
    if clip_db.analysis_status:
        is_analyzed = clip_db.analysis_status.is_fully_analyzed()
        print(f"DB says is_fully_analyzed: {is_analyzed}")
    
    if clip_db.colors:
        analysis_data["color"] = "EXISTS"
    
    # ... (other checks omitted for brevity, focusing on core logic)
    
    needs_analysis_ui_logic = not analysis_data or getattr(clip_db, "needs_reanalysis", True)
    print(f"UI Logic 'needs_analysis' (Before Run): {needs_analysis_ui_logic}")
    print(f"  -> analysis_data populated: {bool(analysis_data)}")
    print(f"  -> needs_reanalysis flag: {getattr(clip_db, 'needs_reanalysis', True)}")


    # 3. Simulate Analysis Run (mocking the results to ensure we test the SAVING logic, not the CV logic)
    # We want to see if _save_analysis_results updates the flags correctly.
    
    print("\n--- Simulating Analysis Save ---")
    analyzer = VideoAnalyzer()
    
    # Mock results typical of a successful run
    mock_results = {
        "clip_id": clip.id,
        "colors": {
            "dominant_colors": [], 
            "temperature": "WARM", 
            "brightness": "BRIGHT",
            "metrics": {} # Analyzer implementation expects this structure sometimes
        },
        "motion": {"motion_type": "STATIC"},
        "scene": {"types": ["INDOOR"]},
        "mood": {"moods": ["CALM"]},
        "objects": {"detected_objects": []},
        "style": {"styles": ["MODERN"]},
        "phash": "12345", 
        "embedding_path": "test.npy"
    }
    
    # Call the internal save method directly to test DB updates
    try:
        analyzer._save_analysis_results(session, clip.id, mock_results)
        
        # Manually update clip needs_reanalysis as analyze_clip does
        clip.needs_reanalysis = False
        session.commit()
        print("Save completed successfully.")
    except Exception as e:
        print(f"Save FAILED: {e}")
        import traceback
        traceback.print_exc()
        
    # 4. Check Post-Analysis State
    print("\n--- Post-Analysis Check ---")
    
    session.expire_all() # Force reload from DB
    clip_final = session.query(VideoClip).filter(VideoClip.id == clip.id).first()
    
    print(f"Final needs_reanalysis: {clip_final.needs_reanalysis}")
    
    status_final = session.query(ClipAnalysisStatus).filter(ClipAnalysisStatus.clip_id == clip.id).first()
    if status_final:
        print(f"Final Analysis Status: Colors={status_final.colors_analyzed}, Scene={status_final.scene_analyzed}")
        print(f"Final is_fully_analyzed(): {status_final.is_fully_analyzed()}")
    
    if not status_final.is_fully_analyzed():
        print("FAILURE: Clip is NOT marked as fully analyzed in DB (ClipAnalysisStatus.is_fully_analyzed() returned False).")
        print(f"  Missing flags: {[k for k,v in {'colors': status_final.colors_analyzed, 'motion': status_final.motion_analyzed, 'scene': status_final.scene_analyzed, 'mood': status_final.mood_analyzed, 'objects': status_final.objects_analyzed, 'style': status_final.style_analyzed, 'fingerprint': status_final.fingerprint_created, 'vector': status_final.vector_extracted}.items() if not v]}")
    else:
        print("SUCCESS: Clip is marked as fully analyzed in DB.")

if __name__ == "__main__":
    debug_analysis()
