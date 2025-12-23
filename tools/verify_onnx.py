import sys
try:
    print("Attempting to import onnxruntime...")
    import onnxruntime as ort
    print(f"SUCCESS: onnxruntime {ort.__version__} imported.")
    print(f"Providers: {ort.get_available_providers()}")
    
    print("\nAttempting to import audio_separator...")
    from audio_separator.separator import Separator
    print("SUCCESS: audio_separator imported.")
    
except ImportError as e:
    print(f"\nCRITICAL ERROR: {e}")
    if "DLL load failed" in str(e):
        print("\nDIAGNOSIS: Missing Visual C++ Redistributable.")
        print("SOLUTION: Please install vc_redist.x64.exe from Microsoft.")
    sys.exit(1)
except Exception as e:
    print(f"\nGenerall Error during import: {e}")
    sys.exit(1)
