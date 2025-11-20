# Real-Time Biosignal User-State Pipeline

This project implements a full pipeline that goes from **raw wrist biosignals** to **real‑time user-state predictions**, with:

- A PLUX / OpenSignals → LSL **physio bridge**
- A PsychoPy **baseline guidance screen**
- A real-time **baseline-aware ML model**
- Multiple **LSL output streams** for Unreal Engine or other clients

---

## 🔌 Quickstart

These steps assume we already have Python and the required libraries installed.

### 1. Clone / copy the project into VS Code

Open the project folder in VS Code so all the `.py` files are available.

### 2. Install dependencies (once)

```bash
pip install numpy scipy pandas scikit-learn joblib pylsl psychopy
```

Optionally add anything else our environment needs (e.g. PLUX/OpenSignals tools).

### 3. Prepare WESAD features and model (offline)

If we have not already done this:

```bash
# 1) Extract window-level baseline-relative features from WESAD
python preprocess_wesad_rel.py

# 2) Train the RandomForest user-state model on those features
python TrainStateModel_WESAD_rel.py
```

That will create:

- `Data/wesad_features_rel.csv`  
- `Models/user_state_model.pkl`

### 4. Start the real-time pipeline

With our PLUX / OpenSignals system ready:

```bash
python RunRealtime.py
```

This will:

1. Look for an LSL stream of type `Physio` (from `BiosignalsLSL.py`).  
2. Launch `BiosignalsLSL.py` automatically if needed (depending on `AUTO_START_BRIDGE`).  
3. Launch `INTRO_BASELINE.py` (PsychoPy baseline UI).  
4. Run a guided baseline phase, then continuous live predictions.  
5. Publish LSL streams:
   - `UserState`  
   - `UserStateProba`  
   - `UserStateStatus`  
   - `UserStateBaselineProgress`  

Unreal Engine or any other client just needs to subscribe to those LSL streams.

---

## 1. Repository Overview

### Offline (data → model)

#### `preprocess_wesad_rel.py`

- Loads WESAD wrist data (EDA + BVP) for each subject.
- Extracts 5‑second sliding-window features with 50% overlap.
- Computes per-subject **baseline means** using windows labeled `baseline`.
- Adds relative features for each absolute feature:

  - `hr_mean_rel`, `hrv_rmssd_rel`  
  - `eda_mean_rel`, `eda_std_rel`, `eda_deriv_std_rel`

- Saves a single CSV:

  - `Data/wesad_features_rel.csv`

Columns include:

- `timestamp`, `subject`, `label`  
- Absolute features: `hr_mean`, `hrv_rmssd`, `eda_mean`, `eda_std`, `eda_deriv_std`  
- Relative features: `*_rel` versions of the above.

#### `TrainStateModel_WESAD_rel.py`

- Loads `Data/wesad_features_rel.csv`.
- Uses only the **relative** features by default:

  - `hr_mean_rel`, `hrv_rmssd_rel`, `eda_mean_rel`, `eda_std_rel`, `eda_deriv_std_rel`

- Sets up a scikit‑learn `Pipeline`:

  - `StandardScaler` → `RandomForestClassifier`

- Uses **GroupKFold** (5 folds) with `subject` as the group to simulate **new subjects**.
- Prints cross‑validated accuracy (mean ± std).
- Trains on all data and saves the final pipeline to:

  - `Models/user_state_model.pkl`

This model file is what we load in `RunRealtime.py`.

---

### Online (real-time)

#### `BiosignalsLSL.py`

- Connects to a PLUX / OpenSignals LSL input stream.
- Reads a multi-channel sample, takes specific indices for:

  - BVP (`BVP_CH`)
  - EDA (`EDA_CH`)

- Maintains circular buffers for BVP, EDA, and timestamps.
- Estimates heart rate from BVP peaks with a simple local-maximum + threshold rule.
- Smooths EDA with a low-pass Butterworth filter.
- Publishes an outgoing LSL stream:

  - **Name**: `BioBridge`  
  - **Type**: `Physio`  
  - **Channels**:
    1. BVP
    2. EDA (filtered)
    3. HR (bpm)

This is the Physio stream `RunRealtime.py` expects.

#### `INTRO_BASELINE.py`

- Opens a full-screen PsychoPy window with:

  - A central geometric shape that changes every few seconds.
  - A baseline instruction text (“please remain seated, avoid movement…”).
  - A horizontal progress bar at the bottom.

- Tries to connect to two LSL streams:

  - `UserStateStatus` (string)  
  - `UserStateBaselineProgress` (float)

- If both are found, uses them to:

  - Drive the progress bar with real baseline progress.
  - Detect when baseline is `calibrating` vs `ready`.

- If not found, falls back to a local timer for a fixed baseline duration.

Controls:

- During baseline: `ESC` → abort immediately.  
- After baseline complete: `SPACE` or `ESC` → exit.

Run alone for testing:

```bash
python INTRO_BASELINE.py
```

#### `RunRealtime.py`

This is the main orchestration script. It:

1. **Loads the trained model** (`Models/user_state_model.pkl`).  
2. **Finds/starts the Physio stream**:
   - Searches for an LSL stream with `type() == "Physio"`.
   - If none is found and `AUTO_START_BRIDGE=True`, launches `BiosignalsLSL.py` and waits.  
3. **Creates four LSL output streams** via `mk_outlets(model)`:
   - `UserState` – 1 string channel (smoothed label).  
   - `UserStateProba` – N float channels (one per model class, labeled in stream metadata).  
   - `UserStateStatus` – 1 string channel for the system status: `waiting`, `calibrating`, `ready`, `stopping`.  
   - `UserStateBaselineProgress` – 1 float channel in [0,1] during baseline.  
4. **Launches `INTRO_BASELINE.py`** in a separate process if the file is present.
5. **Runs baseline collection**:
   - Sends `UserStateStatus="waiting"` initially.  
   - Then sends `UserStateStatus="calibrating"`.  
   - Collects BVP + EDA samples in sliding 5‑second windows with 50% overlap.  
   - For each window, calls `extract_features(...)` and stores the result in `baseline_windows`.  
   - Tracks elapsed time, and publishes `UserStateBaselineProgress = elapsed / BASELINE_SECONDS`.  
   - Stops baseline when:
     - `elapsed >= BASELINE_SECONDS`, and  
     - `len(baseline_windows) >= BASELINE_MIN_WINDOWS`.  
6. **Computes baseline means** from all baseline windows using `compute_baseline_means(...)`.
7. **Enters live prediction mode**:
   - Sends `UserStateStatus="ready"`.  
   - Continues reading BVP + EDA, forms windows, and computes absolute features.  
   - Converts them to baseline-relative features via `rel_from_abs(abs_feats, baseline_means)`.  
   - Builds a feature vector `xvec` from relative or absolute features depending on `USE_RELATIVE_FEATURES_FOR_MODEL`.  
   - Calls `model.predict(xvec)` and (if available) `model.predict_proba(xvec)`.  
   - Smooths predictions over the last `SMOOTH_K` windows with a `deque` and `Counter` majority vote.  
   - Prints predictions and probabilities to the console.  
   - Sends:
     - `UserState = smoothed_label`  
     - `UserStateProba = probabilities` (if available)  
8. **Graceful shutdown**:
   - On `Ctrl+C`, breaks out of the loop.  
   - Sends `UserStateStatus="stopping"`.  
   - Prints a clean shutdown message.

#### `BiosignalsML_baseline.py` (optional standalone live ML)

- Provides a baseline-aware live prediction loop that can be used independently.
- Uses keyboard and/or an LSL `Markers` stream for labeling windows.
- Good for testing ML behavior without the full orchestration of `RunRealtime.py`.

---

## 2. System Architecture

### 2.1 Offline pipeline (WESAD → features → model)

```text
WESAD dataset
   │
   └─ preprocess_wesad_rel.py
        │   (windowed features + per-subject baseline means)
        └─ Data/wesad_features_rel.csv
                │
                └─ TrainStateModel_WESAD_rel.py
                        │
                        └─ Models/user_state_model.pkl
```

### 2.2 Online pipeline (device → LSL → model → LSL)

```text
PLUX / biosignal device
   │
   └─ OpenSignals → LSL (raw PLUX stream)
           │
           └─ BiosignalsLSL.py
                 └─ LSL type="Physio" (BVP, EDA, HR)
                        │
                        └─ RunRealtime.py
                             ├─ Baseline phase
                             │    - accumulate feature windows
                             │    - compute baseline_means
                             │    - send UserStateStatus="calibrating"
                             │    - send UserStateBaselineProgress ∈ [0,1]
                             │
                             └─ Live prediction phase
                                  - extract features
                                  - convert to relative features
                                  - model.predict / model.predict_proba
                                  - smooth predictions over last K windows
                                  - send LSL:
                                      * UserState (smoothed string label)
                                      * UserStateProba (probability vector)
                                      * UserStateStatus ("ready" / "stopping")
                                      * (BaselineProgress still available if desired)

INTRO_BASELINE.py (PsychoPy)
   └─ Subscribes to:
        - UserStateStatus
        - UserStateBaselineProgress
      and shows a guided baseline screen with progress bar

Unreal Engine / other client
   └─ Subscribes to:
        - UserState
        - UserStateProba
        - UserStateStatus
        - Physio (BioBridge)
```

---

## 3. Installation & Requirements

- **Python**: 3.8 or newer recommended.

### 3.1 Python packages

Install from PyPI (basic set):

```bash
pip install numpy scipy pandas scikit-learn joblib pylsl psychopy
```

We may also need:

- PLUX / OpenSignals tools, depending on our hardware setup.
- Any extra dependencies required by our environment (e.g., an LSL plugin for Unreal).

---

## 4. WESAD Dataset Preparation

We assume the WESAD dataset is extracted into a directory like:

```text
Data/
  WESAD/
    S2/
      S2.pkl
      ...
    S3/
      S3.pkl
    ...
```

If our WESAD root is different, either:

- Set an environment variable before running scripts:

  ```bash
  export WESAD_ROOT="/path/to/WESAD"
  ```

- Or arrange our files so that `Data/WESAD` exists with subject folders inside.

---

## 5. Offline Preprocessing & Training

### 5.1 Run preprocessing

```bash
python preprocess_wesad_rel.py
```

What it does:

- Iterates over `S*` folders under `WESAD_ROOT`.
- Loads `.pkl` wrist signals (`BVP`, `EDA`) for each subject.
- Applies 5‑second sliding windows with 50% overlap.
- For each window:
  - Computes HR and HRV from BVP peaks.
  - Smooths EDA with a low-pass filter and computes:
    - `eda_mean`, `eda_std`, `eda_deriv_std`.
  - Aligns with WESAD labels, using majority vote over the window.
  - Maps raw labels:
    - `1 → "baseline"`
    - `2 → "stress"`
    - `3 → "relax"` (amusement treated as relax)
- For each subject, computes baseline means over all `label == "baseline"` windows.
- Adds baseline-relative features for each feature (`*_rel`).
- Drops unlabeled windows (`"unlabeled"`).
- Writes `Data/wesad_features_rel.csv`.

### 5.2 Train the model

```bash
python TrainStateModel_WESAD_rel.py
```

This script:

1. Loads `Data/wesad_features_rel.csv`.  
2. Selects relative features:

   ```text
   hr_mean_rel, hrv_rmssd_rel, eda_mean_rel, eda_std_rel, eda_deriv_std_rel
   ```

3. Uses `GroupKFold(n_splits=5)` with `subject` as the grouping key.  
4. Computes balanced class weights using `compute_class_weight`.  
5. Builds a `Pipeline` with:

   - `StandardScaler()`  
   - `RandomForestClassifier(n_estimators=400, max_depth=10, class_weight=balanced_weights)`  

6. Prints GroupKFold accuracy (mean ± std).  
7. Fits the final model on all data.  
8. Saves it to `Models/user_state_model.pkl`.  
9. Prints a simple train-on-train classification report (for sanity checking only).

---

## 6. Online Components in Detail

### 6.1 BiosignalsLSL.py

- Connects to a PLUX LSL stream (OpenSignals).  
- Reads raw samples, extracts BVP + EDA channels specified by `BVP_CH`, `EDA_CH`.  
- Maintains `deque` buffers for a small sliding window of data.  
- Detects BVP peaks with a local 3-point peak detector and dynamic threshold.  
- Estimates instantaneous HR from RR intervals.  
- Smooths EDA with `lowpass_safe` (Butterworth low-pass + filtfilt when enough samples available).  
- Publishes a 3-channel LSL stream: `[BVP, EDA_filtered, HR]` at ~10 Hz under the name `BioBridge` type `Physio`.

### 6.2 INTRO_BASELINE.py

- Uses PsychoPy to create a minimal but clear baseline screen.  
- Shows geometric shapes, instruction text, and a progress bar.  
- Tries to read:
  - `UserStateStatus`: to transition from calibrating to ready.  
  - `UserStateBaselineProgress`: to update progress bar.  
- If those streams do not exist, falls back to a fixed-duration local timer.  
- Exits when baseline is done and the user presses `SPACE` or `ESC`.

### 6.3 RunRealtime.py (orchestrator)

The main duties are described in the Quickstart and overview sections above. This is the script we run to glue everything together: Physio input, baseline calibration, model evaluation, and LSL output.

---

## 7. Integration with Unreal Engine or Other Clients

Unreal (or any other LSL-capable client) can subscribe to:

- **UserState** (string) – the smoothed discrete label (e.g., `baseline`, `stress`, `relax`).  
- **UserStateProba** (float[N]) – per-label probabilities (order given by the stream’s channel metadata).  
- **UserStateStatus** (string) – system-level state for controlling UI:
  - `waiting`
  - `calibrating`
  - `ready`
  - `stopping`
- **BioBridge** (`Physio`) – raw BVP + EDA + HR, if we want to visualize or process signals directly in Unreal.

We have used it for:

- Driving an avatar’s animation based on `UserState`.  

## 8. Known Assumptions and Notes

- The incoming Physio stream has `type() == "Physio"` and contains BVP + EDA (and optionally HR).  
- Channel indices in `BiosignalsLSL.py` and `RunRealtime.py` are set correctly for our PLUX/OpenSignals setup.  
- WESAD dataset layout matches the original structure (subject folders `S2..S17` with `Sx.pkl` files).  
- PsychoPy and LSL are installed into the same Python environment as the rest of the code.  
- Model training used baseline-relative features if `USE_RELATIVE_FEATURES_FOR_MODEL=True`.

