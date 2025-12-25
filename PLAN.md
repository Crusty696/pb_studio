# Plan: PB Studio Finalization

## 1. Project Context
**Project:** PB Studio (Local AI Video Generation/Editing)
**Phase:** Finalization (Final 25%)
**Core Stack:** Python, PyQt6, SQLite, ONNX Runtime (CUDA/DirectML)

## 2. Strategic Objectives
The primary goal is to stabilize the application for release, ensuring hardware compatibility and a seamless user experience.

### A. Backend Stability & Hardware Compatibility
*   **Goal:** Ensure seamless failover between NVIDIA (CUDA) and AMD/Intel (DirectML).
*   **Verification:** Check `src/pb_studio/bootstrapper.py` and `src/pb_studio/ai/` modules.
*   **Database:** Verify integrity of `project.db` schema and migrations.

### B. UI Finalization (PyQt6)
*   **Goal:** Remove all placeholder elements and "TODO" comments from the GUI.
*   **Focus:** Main Window, Preview widgets, and Settings dialogs.
*   **Action:** Audit `src/pb_studio/gui` directory.

### C. Installer & Deployment
*   **Goal:** Functional One-Click Installer for Windows.
*   **Tool:** Inno Setup (`pb_studio.iss`).
*   **Action:** Verify the `.iss` script includes all necessary assets (ONNX runtimes, FFmpeg, etc.).

## 3. Execution Tasks (TODO)

### Analysis & Verification
*   [ ] **Analyze Startup Logic:** Read `src/pb_studio/bootstrapper.py` to confirm hardware detection logic.
*   [ ] **Database Check:** Run `check_db_status.py` to verify current schema state.
*   [ ] **Dependency Audit:** Check `pyproject.toml` vs `install_pb_studio.ps1` for consistency.

### Implementation & Fixes
*   [ ] **Fix UI Placeholders:** Search for `# TODO` in `src/pb_studio/gui/` and resolve visual gaps.
*   [ ] **Update README:** Replace the incorrect "Desktop Commander" `README.md` with PB Studio documentation.
*   [ ] **Hardware Test Script:** Create/Update a script to explicitly test ONNX provider loading.

### Packaging
*   [ ] **Review Installer:** Audit `pb_studio.iss` paths and inclusion lists.
*   [ ] **Smoke Test:** Run `tests/test_smoke.py`.

## 4. Hardware Support Strategy (Strict)
*   **NVIDIA:** `onnxruntime-gpu`
*   **AMD/Intel:** `onnxruntime-directml`
*   *Code must explicitly handle the switch based on detected hardware.*
