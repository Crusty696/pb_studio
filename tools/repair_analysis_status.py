
import sys
import os
import datetime

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from pb_studio.database.connection import get_db_session
from pb_studio.database.models import VideoClip
from pb_studio.database.models_analysis import ClipAnalysisStatus, ClipFingerprint

def repair_db():
    print("--- Starting Analysis Status Repair ---")
    session = get_db_session()
    
    try:
        clips = session.query(VideoClip).all()
        print(f"Found {len(clips)} clips.")
        
        fixed_count = 0
        
        for clip in clips:
            status = session.query(ClipAnalysisStatus).filter(ClipAnalysisStatus.clip_id == clip.id).first()
            if not status:
                continue
                
            # Check if fingerprint data exists
            fingerprint = session.query(ClipFingerprint).filter(ClipFingerprint.clip_id == clip.id).first()
            has_fingerprint = fingerprint is not None and (fingerprint.phash is not None or fingerprint.dhash is not None)
            
            # Check if vector data exists (we assume if status says vector_extracted OR if there's an embedding entry)
            # ClipFingerprint also holds vector_file
            has_vector = fingerprint is not None and fingerprint.vector_file is not None
            
            # Update status flags if they are False but data exists
            updated = False
            
            if has_fingerprint and not status.fingerprint_created:
                status.fingerprint_created = True
                updated = True
                
            # If vector is marked as false but file exists, fix it
            if has_vector and not status.vector_extracted:
                status.vector_extracted = True
                updated = True
                
            if updated:
                fixed_count += 1
                print(f"Fixed status for clip {clip.id} ({clip.name})")
                
        if fixed_count > 0:
            session.commit()
            print(f"\nSuccessfully repaired {fixed_count} clips.")
        else:
            print("\nNo clips needed repair.")
            
    except Exception as e:
        print(f"Error during repair: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    repair_db()
