# =============================================================================
# preprocess_wesad_rel.py — Extract window-level features from WESAD subjects
#                           and normalize them relative to per-subject baseline
# =============================================================================
#
# GOAL:
#   - Convert raw WESAD wrist data (BVP + EDA) into a CSV of features.
#   - For each subject, compute their *baseline mean features* using
#       windows labeled "baseline".
#   - Produce both:
#       * absolute features  (hr_mean, eda_mean, etc.)
#       * relative features  (feature - subject_baseline_mean)
#
# WHY RELATIVE?
#   Physiological signals vary wildly between people. Subtracting each subject’s
#   baseline produces "how far away from baseline" features, which typically
#   generalize better for ML models.
#
# INPUT:
#   - WESAD dataset directory (S2 … S17 folders with *.pkl data files)
#
# OUTPUT:
#   Data/wesad_features_rel.csv
#
# FEATURES PER WINDOW:
#   hr_mean, hrv_rmssd,
#   eda_mean, eda_std, eda_deriv_std,
#   and *_rel versions of all the above.
#
# WINDOWING:
#   - Window size: 5 seconds
#   - Overlap: 50%
#   - BVP sampled at 64 Hz
#   - EDA sampled at 4 Hz
#
# This script is designed to run once and generate a clean, ML-ready CSV.
# =============================================================================

import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# WESAD root directory — can be supplied via environment variable,
# otherwise defaults to Data/WESAD relative to project root.
WESAD_ROOT = os.environ.get("WESAD_ROOT", str(Path("Data") / "WESAD"))

# Output directory and CSV location.
OUT_DIR = Path("Data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "wesad_features_rel.csv"

# WESAD sampling rates:
FS_BVP = 64.0      # BVP from Empatica E4 wrist
FS_EDA = 4.0       # EDA from Empatica E4 wrist

# Window parameters:
WIN_SEC = 5.0       # 5-second windows
OVERLAP = 0.5        # 50% overlap → step = 2.5 seconds

# For BVP peak detection (minimum spacing between peaks)
DIST_PEAK_SEC = 0.5  # 0.5 seconds = 120 bpm max

# Mapping WESAD integer labels → semantic categories.
LABEL_MAP = {
    1: "baseline",
    2: "stress",
    3: "relax"       # WESAD uses "amusement" (label 3); here we treat it as relax
}


# =============================================================================
# 2. FILTERING HELPERS
# =============================================================================

def butter_lowpass(sig, fs, cutoff=0.5, order=2):
    """
    Safely apply a low-pass filter to EDA.
    - Removes high-frequency noise while preserving slow EDA dynamics.
    - If the signal is too short for filtfilt, returns unfiltered.
    """
    sig = np.asarray(sig, float)

    # Normalized cutoff frequency.
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")

    # filtfilt requires a minimum padding length.
    padlen = 3 * (max(len(a), len(b)) - 1)
    if sig.size <= padlen:
        return sig

    try:
        return filtfilt(b, a, sig)
    except Exception:
        # If filtering fails (rare), just return original
        return sig


def hr_hrv_from_bvp(bvp, fs_bvp=FS_BVP, dist_peak_sec=DIST_PEAK_SEC):
    """
    Extract heart rate (HR) and RMSSD (a time-domain HRV measure) from BVP:
      - Demean BVP
      - Detect peaks using scipy.signal.find_peaks
      - Convert peak distances → RR intervals → HR (bpm)
      - RMSSD computed from successive RR differences

    Returns:
        hr_array   (multiple HR values per interval)
        rmssd      (float, or nan)
    """
    x = np.asarray(bvp, float).squeeze()

    # Flatten if needed
    if x.ndim != 1:
        try: x = x.reshape(-1)
        except Exception: return np.array([]), np.nan

    # Remove NaNs
    x = x[np.isfinite(x)]

    # Need at least ~2 seconds to detect heartbeats reliably
    if x.size < int(fs_bvp * 2):
        return np.array([]), np.nan

    # Remove DC offset
    x = x - np.mean(x)
    x_std = np.std(x)

    if x_std <= 1e-6:
        return np.array([]), np.nan

    # Peak detection rules
    distance = max(1, int(dist_peak_sec * fs_bvp))  # minimum spacing in samples
    prominence = max(0.1 * x_std, 1e-3)

    try:
        peaks, _ = find_peaks(x, distance=distance, prominence=prominence)
    except Exception:
        return np.array([]), np.nan

    # Need at least 3 peaks → at least two RR intervals
    if peaks.size < 3:
        return np.array([]), np.nan

    # RR intervals in seconds
    rr = np.diff(peaks) / fs_bvp
    rr = rr[rr > 0]       # keep only valid intervals

    if rr.size < 2:
        return np.array([]), np.nan

    # HR = 60 / RR (beats per minute)
    hr = 60.0 / rr

    # RMSSD = sqrt(mean(diff(RR)^2))
    rmssd = float(np.sqrt(np.mean(np.square(np.diff(rr))))) if rr.size > 1 else np.nan

    return hr, rmssd


# =============================================================================
# 3. PER-SUBJECT FEATURE EXTRACTION
# =============================================================================

def extract_features_for_subject(subj_path, subj_id):
    """
    Load one subject folder (S2 … S17), extract sliding-window features
    from wrist signals, and return them as a list of dicts.

    Each dict contains:
        timestamp, subject, label,
        hr_mean, hrv_rmssd,
        eda_mean, eda_std, eda_deriv_std
    """

    # WESAD wrist signal file inside Sx/
    pkl_file = next(subj_path.glob("*.pkl"), None)
    if pkl_file is None:
        return []

    # Load the pickle file produced by WESAD authors.
    with open(pkl_file, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    # Extract wrist channels
    wrist = data["signal"]["wrist"]
    labels = np.asarray(data["label"]).astype(int)

    # Raw signals
    eda = np.asarray(wrist["EDA"], float)
    bvp = np.asarray(wrist["BVP"], float)

    # Smooth EDA slightly before windowing
    eda_f = butter_lowpass(eda, FS_EDA, cutoff=0.5, order=2)

    # Compute window lengths in samples for BVP/EDA independently
    win_bvp = int(WIN_SEC * FS_BVP)
    step_bvp = int(max(1, win_bvp * (1 - OVERLAP)))

    win_eda = int(WIN_SEC * FS_EDA)
    step_eda = int(max(1, win_eda * (1 - OVERLAP)))
    
    # Scaling factor to convert EDA indices → label indices
    label_scale = len(labels) / max(1, len(eda_f))

    rows = []

    # Sliding window over EDA (4 Hz)
    for i0 in range(0, len(eda_f) - win_eda + 1, step_eda):
        i1 = i0 + win_eda

        # Window of EDA
        eda_seg = eda_f[i0:i1]

        # Convert EDA indices to BVP indices (64 Hz)
        b0 = int(i0 * (FS_BVP / FS_EDA))
        b1 = b0 + win_bvp
        if b1 > len(bvp):
            break

        bvp_seg = bvp[b0:b1]

        # -------------------------
        # HEART RATE + HRV
        # -------------------------
        hr_vals, rmssd = hr_hrv_from_bvp(bvp_seg, FS_BVP, DIST_PEAK_SEC)
        hr_mean = float(np.nanmean(hr_vals)) if hr_vals.size else np.nan
        hrv_rmssd = float(rmssd) if np.isfinite(rmssd) else np.nan

        # -------------------------
        # EDA FEATURES
        # -------------------------
        eda_arr = np.asarray(eda_seg, float).ravel()
        eda_mean = float(np.mean(eda_arr)) if eda_arr.size else np.nan
        eda_std  = float(np.std(eda_arr)) if eda_arr.size else np.nan

        # First-order derivative → how fast EDA is changing
        if eda_arr.size > 1:
            d = np.diff(eda_arr)
            eda_deriv_std = float(np.std(d))
        else:
            eda_deriv_std = 0.0

        # -------------------------
        # LABEL MAPPING PER WINDOW
        # -------------------------
        # Convert EDA window boundaries into label indices
        L0 = int(i0 * label_scale)
        L1 = int(i1 * label_scale)

        # Clamp + validate
        if L1 <= L0 or L0 >= len(labels):
            lbl_raw = 0
        else:
            L1 = min(L1, len(labels))
            lbl_slice = labels[L0:L1]

            # Majority vote of labels inside this window
            if lbl_slice.size > 0:
                vals, counts = np.unique(lbl_slice, return_counts=True)
                lbl_raw = int(vals[np.argmax(counts)])
            else:
                lbl_raw = 0

        # Convert raw integer label → string category
        label = LABEL_MAP.get(lbl_raw, "unlabeled")

        # -------------------------
        # SAVE WINDOW FEATURES
        # -------------------------
        rows.append({
            "timestamp": i0 / FS_EDA,   # seconds from start
            "subject": subj_id,
            "label": label,
            "hr_mean": hr_mean,
            "hrv_rmssd": hrv_rmssd,
            "eda_mean": eda_mean,
            "eda_std": eda_std,
            "eda_deriv_std": eda_deriv_std
        })

    return rows


# =============================================================================
# 4. MAIN PIPELINE: PER-SUBJECT + BASELINE NORMALIZATION
# =============================================================================

def main():
    root = Path(WESAD_ROOT)
    if not root.exists():
        raise SystemExit(f"WESAD root not found: {root}")

    all_rows = []

    # Each subject folder is named S2 … S17
    for subj in sorted(root.glob("S*")):

        rows = extract_features_for_subject(subj, subj.name)
        if not rows:
            print(f"{subj.name}: no windows extracted")
            continue

        df_s = pd.DataFrame(rows)

        # -------------------------
        # PER-SUBJECT BASELINE
        # -------------------------
        base = df_s[df_s["label"] == "baseline"]

        if base.empty:
            print(f"{subj.name}: WARNING → no baseline rows; "
                  "relative features will be set to 0 for this subject")

            # No baseline → define relative features as zero (neutral)
            for col in ["hr_mean","hrv_rmssd","eda_mean","eda_std","eda_deriv_std"]:
                df_s[col + "_rel"] = 0.0

        else:
            # Compute baseline means across all baseline windows
            featcols = ["hr_mean","hrv_rmssd","eda_mean","eda_std","eda_deriv_std"]
            bmeans = base[featcols].mean()

            # Relative = absolute - subject_baseline_mean
            for col in featcols:
                df_s[col + "_rel"] = df_s[col] - float(bmeans[col])

        # Report label counts for debugging
        print(f"{subj.name} label counts:", df_s["label"].value_counts().to_dict())

        all_rows.append(df_s)

    # -------------------------
    # CONCATENATE ALL SUBJECTS
    # -------------------------
    if not all_rows:
        print("No features extracted from any subject; aborting.")
        return

    df = pd.concat(all_rows, ignore_index=True)

    # Drop windows with no valid label
    df = df[df["label"] != "unlabeled"]

    # Save final CSV
    df.to_csv(OUT_CSV, index=False)

    print(f"Saved {len(df)} labeled feature rows with relative normalization to:")
    print(f"  {OUT_CSV.resolve()}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
