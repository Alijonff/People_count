# Implementation Plan - People Counter Restructuring & GPU Fix

## Goal
Restructure the `People_count` project into two distinct logic paths as requested, and ensure all models (YOLO + InsightFace/FaceRecognition) run on GPU.

## User Review Required
> [!IMPORTANT]
> I will be reinstalling PyTorch. This might take a few minutes and require internet access.
> I will delete the old python files in the root directory after moving the logic to the new folders.

## Proposed Changes

### 1. GPU Fix
- Uninstall current `torch`, `torchvision`, `torchaudio`.
- Install PyTorch with CUDA 12.4 support (compatible with CUDA 12.6 drivers).
- Verify with `check_gpu.py`.

### 2. Project Structure
Create two new directories:
- `logic_known_employees/`
- `logic_dynamic_tracking/`

### 3. Logic 1: Known Employees (`logic_known_employees/`)
- **`register_face.py`**: [NEW] Script to capture a face from webcam, ask for a name, and save embedding to `face_db`.
- **`main_known.py`**: [NEW] Based on `people_counter_faceid.py`.
    - Loads `face_db`.
    - Detects people.
    - Matches faces to `face_db`.
    - If match: Log time for that Employee.
    - If no match: Label as "Unknown" (do not track time or track as Unknown).

### 4. Logic 2: Dynamic Tracking (`logic_dynamic_tracking/`)
- **`main_dynamic.py`**: [NEW] Based on `people_counter_facecentric.py`.
    - Starts with empty DB.
    - Detects people.
    - If face seen:
        - Compare with existing dynamic IDs.
        - If match: Update that ID.
        - If no match: Create new ID (Person_1, Person_2...).
    - Persist these IDs during the session (and optionally save to disk if restart needed, but "session-based" is implied for now).

### 5. Cleanup
- [DELETE] `people_counter.py`
- [DELETE] `people_counter_facecentric.py`
- [DELETE] `people_counter_faceid.py`
- [DELETE] `people_counter_first_seen.py`
- [DELETE] `face_db.py` (will be moved/adapted inside folders or kept as common util if preferred, but user asked to separate completely, so I will duplicate utils to keep folders self-contained).

## Verification Plan
### Automated Tests
- Run `check_gpu.py` to confirm `CUDA available: True`.

### Manual Verification
- **Logic 1**: Run registration, add myself. Run main, check if it recognizes me.
- **Logic 2**: Run main, show face, leave frame, come back. Check if ID persists.
