# PB Studio - Production Readiness Roadmap

This document outlines the necessary steps to transition PB Studio from its current state to a production-ready application.

**Current Status:** Pre-Alpha / Experimental
**Verification Status:** FAILED (Audit 2024-12-20)

## Phase 1: Stabilization & Infrastructure (P0)

*Goal: Ensure the codebase builds, runs, and tests reliably on a clean environment.*

- [ ] **Dependency Consolidation**:
    - [ ] Audit `pyproject.toml`. It currently lists optional dependencies that are required at runtime (e.g., `librosa`, `scikit-image` via `skimage`).
    - [ ] Hard-pin critical versions to prevent "it works on my machine" issues.
    - [ ] Create a `requirements.txt` freeze for the exact verified environment.

- [ ] **Environment Setup**:
    - [ ] Create a robust `Dockerfile` or `setup_env.sh` that installs system libraries (ffmpeg, libsndfile) which are currently manual prerequisites.
    - [ ] Fix `python-magic` installation issues on Windows/Linux (detected in logs).

- [ ] **Continuous Integration (CI)**:
    - [ ] Set up GitHub Actions (or similar) to run tests on push.
    - [ ] Enforce "No Binary Commit" policy (exclude `*.db`, `*.pyc`, `cache/`).

## Phase 2: Testing & Quality Assurance (P1)

*Goal: Move from 0% to >50% code coverage to prevent regression.*

- [ ] **Unit Tests**:
    - [ ] `src/pb_studio/database`: Test connection pooling, migration, and CRUD models.
    - [ ] `src/pb_studio/pacing`: Test the `AdvancedPacingEngine` logic without requiring full media files (mocking).
    - [ ] `src/pb_studio/video`: Test video analysis parsers.

- [ ] **Integration Tests**:
    - [ ] Convert `test_full_render.py` into a proper `pytest` suite using fixtures for dummy audio/video.
    - [ ] Verify database state persistence across restarts.

## Phase 3: Technical Debt & Refactoring (P2)

*Goal: Address "TODOs" and "FIXMEs" left in the code.*

- [ ] **Configuration Management**:
    - [ ] `src/pb_studio/utils/gpu_memory.py`: Integrate with central `Config` manager (currently TODO).
    - [ ] Remove hardcoded paths in `test_full_render.py` (partially fixed, needs polish).

- [ ] **Pacing Engine Improvements**:
    - [ ] `src/pb_studio/pacing/ai_enhanced_pacing_engine.py`: Implement actual scene transition detection (currently TODO).
    - [ ] `src/pb_studio/pacing/structure_analyzer.py`: Implement energy gradient detection for better structure analysis.
    - [ ] `src/pb_studio/video/video_compatibility_manager.py`: Implement keyframe detection logic.

- [ ] **Database Hygiene**:
    - [ ] Remove hardcoded `data/project.db` paths; allow environment variable override for testing.
    - [ ] Implement proper schema migration handling (Alembic is likely needed if not fully utilized).

## Phase 4: Release Preparation (P3)

- [ ] **Packaging**:
    - [ ] Create PyInstaller spec files for Windows and Linux.
    - [ ] Verify `Start-PBStudio.ps1` and `Start_PB_Studio.bat` match the new dependency structure.

- [ ] **Documentation**:
    - [ ] Update `README.md` with correct installation instructions (removing "poetry optional" confusion).
    - [ ] Document the "CutList" data model changes.

## Immediate Next Steps (Task List)

1. [ ] Fix `pyproject.toml` to include `librosa` and `scikit-image` as core dependencies.
2. [ ] Expand `src/pb_studio/tests/test_models.py` to cover `AudioTrackReference`.
3. [ ] Implement `src/pb_studio/utils/gpu_memory.py` config integration.
