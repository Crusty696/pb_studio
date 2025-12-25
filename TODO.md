# TODO: PB Studio Finalization

## Phase 1: Critical Fixes (Backend Stability)
- [ ] **Fix AMD/DirectML Crash (Stem Separation)**
  - [ ] **Diagnosis:** `DML_VISIBLE_DEVICES` re-indexing conflict.
  - [ ] **Task:** Modify `src/pb_studio/bootstrapper.py` or `src/pb_studio/audio/stem_separator.py`.
    - If `DML_VISIBLE_DEVICES` is set, ensure ONNX Runtime receives `device_id=0`.
  - [ ] **Verification:** Run `tests/test_smoke.py` or manual stem separation test.
- [ ] **Database & Config**
  - [ ] Run `python check_db_status.py` to confirm DB health.
  - [ ] Verify `config.ini` defaults.

## Phase 2: User Interface (UI) Polishing
- [ ] **Code Audit**
  - [ ] **Scan:** Search for `# TODO` and `pass` blocks in `src/pb_studio/gui/`.
  - [ ] **Fix:** Resolve identified UI placeholders (e.g., Settings dialog, Export widget).
- [ ] **Settings UI**
  - [ ] Ensure "Hardware Acceleration" dropdown correctly shows "DirectML" vs "CUDA" based on system.

## Phase 3: Documentation & Cleanup
- [ ] **Documentation**
  - [ ] **CRITICAL:** Overwrite `README.md` (currently "Desktop Commander") with PB Studio documentation.
  - [ ] Keep `interaction_log.md` updated.
- [ ] **Cleanup**
  - [ ] Implement robust `temp/` folder cleanup on exit in `main.py`.

## Phase 4: Packaging & Installer
- [ ] **Installer Script**
  - [ ] Audit `pb_studio.iss` for correct file paths and DLL inclusions.
  - [ ] Ensure `onnxruntime.dll` / `onnxruntime_providers_directml.dll` are packed.
- [ ] **Final Release Build**
  - [ ] Build installer.
  - [ ] Perform final install/uninstall test.