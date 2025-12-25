# Interaction Log - PB Studio Finalization

## 2025-12-24: Initial Health Check & Status Report

### 1. Codebase vs. App Status
*   **Startup/Bootstrapper:** Implemented (`src/pb_studio/bootstrapper.py`). Logic for detecting CUDA vs DirectML exists but is causing a crash on AMD systems due to device index misalignment.
*   **Audio Analysis:** Robust and working. Beat detection, onset detection, and timeline rendering are functional (verified in logs).
*   **Stem Separation:** **BROKEN** on AMD.
    *   **Error:** `Windows fatal exception: access violation` in `onnx.shape_inference`.
    *   **Context:** System detects two GPUs (Integrated + Dedicated RX 7800 XT).
    *   **Diagnosis:** Bootstrapper sets `DML_VISIBLE_DEVICES=1` to force the dedicated GPU. However, this likely re-indexes the devices so that the dedicated GPU becomes Index 0. If the application then tries to use `Index 1` (based on the original detection), it fails or accesses invalid memory.
*   **UI Status:**
    *   Timeline renders correctly.
    *   Audio loading progress bars work.
    *   *To be verified:* Remaining TODOs in GUI code.

### 2. Log Analysis Findings
*   **File:** `logs/stem_separation.log` / `logs/app.log`
*   **Crash Stack Trace:**
    ```text
    File "onnx\shape_inference.py", line 58 in infer_shapes
    File "audio_separator\separator\architectures\mdx_separator.py", line 128 in load_model
    ...
    Windows fatal exception: access violation
    ```
*   **GPU Detection Log:**
    ```text
    INFO | Nutze vom Bootstrapper gesetzte GPU: Unknown (Index 1)
    INFO | DML_VISIBLE_DEVICES=1 gesetzt (dedizierte GPU)
    INFO | StemSeparator initialized: Preset=kuielab, DirectML=True
    ```

### 3. Action Items (Immediate)
1.  **Fix GPU Indexing:** Modify `src/pb_studio/bootstrapper.py` and `src/pb_studio/audio/stem_separator.py`.
    *   **Strategy:** If `DML_VISIBLE_DEVICES` is set, the application must use Device Index `0` for the ONNX Runtime session, as the filtered view only shows that one GPU.
2.  **UI Audit:** Complete the scan for `# TODO` items in `src/pb_studio/gui`.
3.  **Documentation:** Update `README.md` to reflect the actual project (PB Studio).

### 4. Remaining Work (from Plan)
*   [ ] Fix AMD/DirectML Stem Separation Crash.
*   [ ] Resolve GUI TODOs.
*   [ ] Update Installer (`.iss`) and Verification Scripts.
*   [ ] Documentation Cleanup.

