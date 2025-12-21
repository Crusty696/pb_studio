
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))

from pb_studio.database.connection import get_db_session
from pb_studio.database.models import VideoClip
from pb_studio.database.models_analysis import ClipAnalysisStatus

def check_status():
    session = get_db_session()
    clips = session.query(VideoClip).all()
    print(f"Total clips: {len(clips)}")
    
    unanalyzed_flag = 0
    no_analysis_status = 0
    partially_analyzed = 0
    fully_analyzed = 0
    
    for clip in clips:
        if clip.needs_reanalysis:
            unanalyzed_flag += 1
            
        status = session.query(ClipAnalysisStatus).filter(ClipAnalysisStatus.clip_id == clip.id).first()
        if not status:
            no_analysis_status += 1
        elif status.is_fully_analyzed():
            fully_analyzed += 1
        else:
            partially_analyzed += 1
            # Print detail for first few partials
            if partially_analyzed <= 3:
                print(f"Partial Clip {clip.id}: Colors={status.colors_analyzed}, Motion={status.motion_analyzed}, Scene={status.scene_analyzed}, Mood={status.mood_analyzed}, Objects={status.objects_analyzed}, Style={status.style_analyzed}")
                
    print(f"Clips with needs_reanalysis=True: {unanalyzed_flag}")
    print(f"Clips with NO analysis_status record: {no_analysis_status}")
    print(f"Clips partially analyzed: {partially_analyzed}")
    print(f"Clips fully analyzed: {fully_analyzed}")

if __name__ == "__main__":
    check_status()
