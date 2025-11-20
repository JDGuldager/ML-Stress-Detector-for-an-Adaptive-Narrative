# =============================================================================
# RunRealtime.py — One-command pipeline with guided baseline + live predictions + LSL
# =============================================================================
#
# High-level overview
# -------------------
# This script coordinates the whole real-time system:
#
#   1. Ensures there is an incoming LSL "Physio" stream (BVP + EDA).
#      - If missing, it can auto-launch BiosignalsLSL.py to create that stream.
#
#   2. Launches the PsychoPy baseline UI (INTRO_BASELINE.py) in a separate process.
#      - That UI listens to:
#          * "UserStateStatus"            (waiting / calibrating / ready / stopping)
#          * "UserStateBaselineProgress"  (0.0–1.0)
#      - We drive those LSL streams from this script.
#
#   3. Baseline phase:
#      - Reads BVP + EDA from Physio stream.
#      - Extracts features over sliding 5-second windows.
#      - Accumulates those windows as "baseline windows".
#      - Sends status="calibrating" and live progress [0–1] over LSL.
#
#   4. After enough baseline windows are gathered:
#      - Computes per-feature baseline means (per person).
#      - Sends status="ready" over LSL so the PsychoPy screen knows we’re done.
#
#   5. Prediction phase:
#      - Continues reading BVP + EDA.
#      - For each new window, extracts features.
#      - Converts them to baseline-relative features.
#      - Sends predictions (string + probabilities) over LSL:
#            * UserState               (string: smoothed label)
#            * UserStateProba          (float[N]: probabilities)
#            * UserStateStatus         (string: state of this pipeline)
#            * UserStateBaselineProgress (float: 0..1, used only during baseline)
#
# Typical usage:
#   python RunRealtime.py
#
# Everything else (PsychoPy screen, Unreal visualization, etc.) just needs to
# connect to the LSL streams we create here.
# =============================================================================

import os
import sys
import time
import subprocess
from pathlib import Path
from collections import deque, Counter

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from pylsl import StreamInlet, resolve_streams, StreamInfo, StreamOutlet
import joblib


# =============================================================================
# 1. USER / PIPELINE SETTINGS
# =============================================================================

# File path to our trained ML model (RandomForest inside a scikit-learn Pipeline).
# This file is expected to be created by TrainStateModel_WESAD_rel.py.
MODEL_PATH = "Models/user_state_model.pkl"

# If True, and no Physio stream is found, we attempt to start BiosignalsLSL.py automatically.
AUTO_START_BRIDGE = True

# Name of the script that publishes the Physio LSL stream (BVP + EDA).
BRIDGE_SCRIPT = "BiosignalsLSL.py"

# Expected sampling rate (Hz) of the incoming Physio stream.
FS = 10

# Channel indices in the incoming Physio stream:
#   BVP_CH = index of BVP channel
#   EDA_CH = index of EDA channel
BVP_CH, EDA_CH = 0, 1

# Feature window length (seconds).
WINDOW_SEC = 5

# Fractional overlap between consecutive windows.
#   0.5 means: each new window starts 2.5 seconds after the previous one.
OVERLAP = 0.5

# Cutoff frequency (Hz) for low-pass filtering EDA before computing statistics.
EDA_LP = 0.5

# Prediction smoothing:
#   We keep the last SMOOTH_K predicted labels and take the majority vote.
SMOOTH_K = 3

# Baseline collection parameters:
#   - We aim for BASELINE_SECONDS of data for the participant to relax.
#   - BASELINE_MIN_WINDOWS enforces that we have at least that many windows.
BASELINE_SECONDS = 60
BASELINE_MIN_WINDOWS = 8

# Whether our trained model expects relative features ( *_rel ) or absolute ones.
USE_RELATIVE_FEATURES_FOR_MODEL = True

# Name of the PsychoPy script used for the visual baseline instructions.
INTRO_SCRIPT_NAME = "INTRO_BASELINE.py"


# =============================================================================
# 2. SMALL PRINT HELPERS (CONSISTENT LOGGING)
# =============================================================================

def info(msg):
    """Print an informational message with a consistent prefix."""
    print(f"[INFO] {msg}")

def warn(msg):
    """Print a warning message with a consistent prefix."""
    print(f"[WARN] {msg}")

def err(msg):
    """Print an error message with a consistent prefix."""
    print(f"[ERR ] {msg}")


# =============================================================================
# 3. SIGNAL PROCESSING HELPERS
# =============================================================================

def butter_lowpass(sig, fs, cutoff=0.5, order=2):
    """
    Apply a low-pass Butterworth filter to a 1D signal `sig`.

    Details:
        - `fs` is the sampling rate in Hz.
        - `cutoff` is the cutoff frequency in Hz.
        - `order` controls filter steepness.

    Safety:
        - filtfilt needs a minimum number of samples (pad length).
        - If `sig` is too short, we skip filtering and just return the raw signal.
    """
    sig = np.asarray(sig, float)

    # Design normalized low-pass filter (cutoff / Nyquist).
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")

    # Required pad length for filtfilt (rough estimate from SciPy docs).
    padlen = 3 * (max(len(a), len(b)) - 1)
    if sig.size <= padlen:
        # Not enough data points → just return the unfiltered array.
        return sig

    try:
        # filtfilt runs filter forward and backward → zero phase shift.
        return filtfilt(b, a, sig)
    except Exception:
        # If something goes wrong numerically, we fall back to the raw signal.
        return sig


def extract_features(bvp, eda, fs):
    """
    Extract features from BVP + EDA for a single window.

    Inputs:
        bvp : array-like, BVP samples in the window.
        eda : array-like, EDA samples in the window.
        fs  : sampling rate in Hz (for BVP; EDA is assumed to use same `fs` here).

    Outputs (dict):
        {
          "hr_mean":      mean heart rate in bpm,
          "hrv_rmssd":    RMSSD (time-domain HRV measure),
          "eda_mean":     mean EDA,
          "eda_std":      standard deviation of EDA,
          "eda_deriv_std":std of first derivative of EDA
        }
    """

    # -----------------------------
    # 3.1 HR / HRV from BVP peaks
    # -----------------------------
    x = np.asarray(bvp, float).ravel()

    # Remove any NaNs/infs that might appear from upstream processing.
    x = x[np.isfinite(x)]

    # Initialize outputs as NaN in case we cannot compute them.
    hr_mean = np.nan
    hrv_rmssd = np.nan

    # Require at least ~2 seconds of data to detect peaks.
    if x.size >= int(fs * 2):
        # Remove DC offset so peaks are centered around zero.
        x = x - np.mean(x)
        x_std = np.std(x)

        if x_std > 1e-6:
            # Minimum peak distance (in samples):
            #   0.5 seconds * fs = half a second between peaks (up to 120 bpm).
            distance = int(max(1, 0.5 * fs))

            # Peak prominence threshold scales with signal standard deviation.
            prominence = max(0.1 * x_std, 1e-3)

            try:
                # Detect peaks that are spaced at least `distance` samples and
                # have at least `prominence` amplitude.
                peaks, _ = find_peaks(x, distance=distance, prominence=prominence)

                # Require at least 3 peaks → at least two RR intervals.
                if peaks.size >= 3:
                    rr = np.diff(peaks) / fs      # RR intervals in seconds.
                    rr = rr[rr > 0]              # drop any non-positive values.

                    if rr.size >= 2:
                        # HR for each RR interval in bpm.
                        hr = 60.0 / rr
                        hr_mean = float(np.nanmean(hr))

                        # RMSSD = sqrt(mean(diff(RR)^2)).
                        hrv_rmssd = float(
                            np.sqrt(np.mean(np.square(np.diff(rr))))
                        )
            except Exception:
                # Any failure in peak detection leaves hr_mean and hrv_rmssd as NaN.
                pass

    # -----------------------------
    # 3.2 EDA smoothing + statistics
    # -----------------------------
    # Smooth EDA to reduce high-frequency noise.
    eda_s = butter_lowpass(eda, fs, cutoff=EDA_LP, order=2)

    # Convert to a flat float array.
    eda_arr = np.asarray(eda_s, float).ravel()

    # Mean and std are straightforward statistics.
    eda_mean = float(np.mean(eda_arr)) if eda_arr.size else np.nan
    eda_std  = float(np.std(eda_arr))  if eda_arr.size else np.nan

    # Derivative: how quickly EDA is changing.
    if eda_arr.size > 1:
        d = np.diff(eda_arr)
        eda_deriv_std = float(np.std(d))
    else:
        eda_deriv_std = 0.0

    return {
        "hr_mean": hr_mean,
        "hrv_rmssd": hrv_rmssd,
        "eda_mean": eda_mean,
        "eda_std": eda_std,
        "eda_deriv_std": eda_deriv_std
    }


# =============================================================================
# 4. BASELINE HANDLING
# =============================================================================

def compute_baseline_means(feature_rows):
    """
    Compute per-feature baseline means from a list of feature dicts.

    Input:
        feature_rows : list of dicts; each dict has keys:
            "hr_mean", "hrv_rmssd", "eda_mean", "eda_std", "eda_deriv_std"

    Output:
        baseline_means : dict mapping each feature name → mean value across rows.

    NaN handling:
        We use np.nanmean so windows with NaN values are ignored for that feature.
    """
    if not feature_rows:
        # Fallback: if baseline_windows is unexpectedly empty, we return NaNs
        # so that relative features become 0.0 later (since we treat NaN baseline
        # as "no offset").
        return {
            k: np.nan
            for k in ["hr_mean", "hrv_rmssd", "eda_mean", "eda_std", "eda_deriv_std"]
        }

    df = pd.DataFrame(feature_rows)

    return {
        k: float(np.nanmean(df[k].values))
        for k in ["hr_mean", "hrv_rmssd", "eda_mean", "eda_std", "eda_deriv_std"]
    }


def rel_from_abs(feats_abs, baseline_means):
    """
    Convert absolute features to baseline-relative features:

        rel_feature = absolute_feature - baseline_mean

    Inputs:
        feats_abs      : dict of absolute features for the current window.
        baseline_means : dict of per-feature baseline mean values.

    Output:
        dict with keys like "hr_mean_rel", "eda_std_rel", etc.

    If the baseline mean is NaN for a feature, we treat the relative feature
    as 0.0 (neutral deviation).
    """
    out = {}
    for k, v in feats_abs.items():
        b = baseline_means.get(k, np.nan)
        if np.isfinite(b):
            out[k + "_rel"] = float(v) - float(b)
        else:
            out[k + "_rel"] = 0.0
    return out


# =============================================================================
# 5. LSL STREAM MANAGEMENT
# =============================================================================

def ensure_physio_stream():
    """
    Locate an LSL stream of type "Physio". If none is found:
        - optionally start BiosignalsLSL.py to create one,
        - wait again and try to discover it.

    Returns:
        A pylsl.StreamInfo object for the Physio stream.

    Raises:
        SystemExit if we fail to find a Physio stream after waiting.
    """
    info("Searching for LSL Physio stream...")
    t0 = time.time()
    stream = None

    # First attempt: search for up to 10 seconds.
    while time.time() - t0 < 10:
        for s in resolve_streams():
            if s.type() == "Physio":
                stream = s
                break
        if stream:
            break
        time.sleep(0.5)

    if stream:
        info(f"Found Physio stream: {stream.name()}")
        return stream

    # If not found and auto-start is enabled, fire up the bridge script.
    if AUTO_START_BRIDGE:
        warn("No Physio stream found. Attempting to start BiosignalsLSL.py...")

        try:
            # Launch BiosignalsLSL.py in a separate process, suppressing its stdout/stderr.
            subprocess.Popen(
                [sys.executable, BRIDGE_SCRIPT],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            err(f"Could not start {BRIDGE_SCRIPT}: {e}")

        info("Waiting for Physio stream (up to 20s)...")
        t0 = time.time()
        while time.time() - t0 < 20:
            for s in resolve_streams():
                if s.type() == "Physio":
                    info(f"Found Physio stream: {s.name()}")
                    return s
            time.sleep(0.5)

    # If we get here, we never found a Physio stream.
    raise SystemExit("Physio stream not available. "
                     "Start BiosignalsLSL.py or set AUTO_START_BRIDGE=False.")


def mk_outlets(model):
    """
    Create and return the LSL outlets used by this script.

    Outlets:
        - UserState (string)           : final (smoothed) label.
        - UserStateProba (float[N])    : probabilities for each label.
        - UserStateStatus (string)     : controls UI state (waiting, calibrating, ready, stopping).
        - UserStateBaselineProgress    : progress [0..1] during baseline.

    Also returns:
        labels : list of label names (order used in UserStateProba).
    """
    # Determine label order from the model:
    #   - If model is a Pipeline with step name "clf", use that.
    #   - Else try model.classes_.
    try:
        labels = list(model.named_steps["clf"].classes_)
    except Exception:
        try:
            labels = list(model.classes_)
        except Exception:
            labels = ["baseline", "stress", "relax"]

    info(f"Model label order: {labels}")

    # --- UserState: final classification label ---
    info_state = StreamInfo(
        "UserState",      # stream name
        "Classification", # stream type
        1,                # 1 channel: just the label string
        0,                # irregular sampling (event-based)
        "string",         # channel format
        "user_state"      # source id
    )
    outlet_state = StreamOutlet(info_state)

    # --- UserStateProba: class probabilities in fixed label order ---
    info_prob = StreamInfo(
        "UserStateProba",     # stream name
        "Probs",              # stream type
        len(labels),          # one channel per label
        0,                    # event-based
        "float32",            # floats
        "user_state_proba"    # source id
    )
    # Annotate the probability stream with label names.
    desc = info_prob.desc().append_child("channels")
    for lbl in labels:
        ch = desc.append_child("channel")
        ch.append_child_value("label", lbl)
    outlet_prob = StreamOutlet(info_prob)

    # --- UserStateStatus: overall status of our pipeline ---
    info_status = StreamInfo(
        "UserStateStatus",   # stream name
        "Status",            # type
        1,
        0,
        "string",
        "user_state_status"
    )
    outlet_status = StreamOutlet(info_status)

    # --- UserStateBaselineProgress: 0..1 progress during baseline ---
    info_progress = StreamInfo(
        "UserStateBaselineProgress",  # stream name
        "Baseline",                   # type
        1,
        0,
        "float32",
        "user_state_baseline_progress"
    )
    outlet_progress = StreamOutlet(info_progress)

    info("Broadcasting LSL streams: UserState, UserStateProba, UserStateStatus, UserStateBaselineProgress")
    return labels, outlet_state, outlet_prob, outlet_status, outlet_progress


# =============================================================================
# 6. TERMINAL PROGRESS BAR (FOR BASELINE)
# =============================================================================

def progress_bar(prefix, elapsed, total):
    """
    Draw a simple ASCII progress bar in the terminal.

    Example:
        Calibrating baseline [##########----------] 50%
    """
    total = max(1, int(total))
    pct = min(1.0, elapsed / total)

    bar_len = 30
    filled = int(pct * bar_len)
    bar = "#" * filled + "-" * (bar_len - filled)

    print(f"\r{prefix} [{bar}] {int(pct * 100)}%", end="", flush=True)


# =============================================================================
# 7. MAIN ENTRY POINT
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # 7.1 Load trained model
    # -------------------------------------------------------------------------
    if not Path(MODEL_PATH).exists():
        raise SystemExit(f"Model not found at {MODEL_PATH}. "
                         "Run the training script first.")

    model = joblib.load(MODEL_PATH)

    # -------------------------------------------------------------------------
    # 7.2 Connect to incoming Physio stream (BVP + EDA)
    # -------------------------------------------------------------------------
    stream = ensure_physio_stream()
    inlet = StreamInlet(stream)

    # -------------------------------------------------------------------------
    # 7.3 Create LSL output streams for state, probabilities, status, progress
    # -------------------------------------------------------------------------
    LABELS, outlet_state, outlet_prob, outlet_status, outlet_progress = mk_outlets(model)

    # -------------------------------------------------------------------------
    # 7.4 Launch PsychoPy baseline UI (INTRO_BASELINE.py), if present
    # -------------------------------------------------------------------------
    try:
        intro_script = Path(INTRO_SCRIPT_NAME)
        if intro_script.exists():
            info(f"Launching PsychoPy baseline UI: {intro_script}")
            subprocess.Popen([sys.executable, str(intro_script)])
        else:
            warn(f"{INTRO_SCRIPT_NAME} not found in the current directory.")
    except Exception as e:
        warn(f"Could not start {INTRO_SCRIPT_NAME}: {e}")

    # -------------------------------------------------------------------------
    # 7.5 Initialize buffers and baseline collections
    # -------------------------------------------------------------------------
    window_samples = int(WINDOW_SEC * FS)                 # window size in samples
    step_samples = int(max(1, window_samples * (1 - OVERLAP)))  # step size in samples

    buf_bvp, buf_eda = [], []                             # raw signal buffers
    pred_buf = deque(maxlen=SMOOTH_K)                     # for prediction smoothing

    baseline_windows = []                                 # list of feature dicts for baseline

    # Before baseline starts, we broadcast "waiting" so the PsychoPy UI can show
    # "waiting for baseline" or similar state.
    try:
        outlet_status.push_sample(["waiting"])
    except Exception:
        pass

    print("\n-----------------------------------------------")
    print("Baseline calibration is about to start.")
    print("PsychoPy baseline window should appear.")
    print("-----------------------------------------------\n")

    # Small pause to let the PsychoPy window open.
    time.sleep(2.0)

    # Now we broadcast that baseline calibration is active.
    try:
        outlet_status.push_sample(["calibrating"])
    except Exception:
        pass

    print("Baseline calibration starting now. Please keep still and relaxed.")
    print(f"Target baseline duration: ~{BASELINE_SECONDS} seconds.\n")

    # -------------------------------------------------------------------------
    # 7.6 BASELINE COLLECTION LOOP
    # -------------------------------------------------------------------------
    t_start = time.time()

    while True:
        # 1) Pull one sample from the Physio stream.
        sample, ts = inlet.pull_sample(timeout=0.1)

        # Append BVP + EDA to buffers if sample is valid and has required channels.
        if sample and len(sample) > max(BVP_CH, EDA_CH):
            buf_bvp.append(sample[BVP_CH])
            buf_eda.append(sample[EDA_CH])

        # 2) As soon as we accumulate at least `window_samples`,
        #    we extract one baseline feature window.
        if len(buf_bvp) >= window_samples and len(buf_eda) >= window_samples:
            bvp_seg = np.array(buf_bvp[-window_samples:])
            eda_seg = np.array(buf_eda[-window_samples:])

            feats_abs = extract_features(bvp_seg, eda_seg, FS)
            baseline_windows.append(feats_abs)

            # Slide window forward by step_samples:
            buf_bvp = buf_bvp[step_samples:]
            buf_eda = buf_eda[step_samples:]

        # 3) Update terminal progress + status LSL for PsychoPy.
        elapsed = time.time() - t_start
        progress_bar("Calibrating baseline", elapsed, BASELINE_SECONDS)

        # Send normalized progress in [0.0, 1.0].
        try:
            frac = max(0.0, min(1.0, elapsed / float(BASELINE_SECONDS)))
            outlet_progress.push_sample([float(frac)])
        except Exception:
            pass

        # 4) Termination condition for baseline phase:
        #      - We have been collecting for at least BASELINE_SECONDS seconds, AND
        #      - We have at least BASELINE_MIN_WINDOWS feature windows.
        if (elapsed >= BASELINE_SECONDS) and (len(baseline_windows) >= BASELINE_MIN_WINDOWS):
            break

    print("\nBaseline collection finished. Computing per-feature baseline means...")

    # Compute baseline means from all baseline windows.
    baseline_means = compute_baseline_means(baseline_windows)
    info(f"Baseline means: {baseline_means}")

    # Notify PsychoPy that baseline is ready (UI typically switches messages).
    try:
        outlet_status.push_sample(["ready"])
    except Exception:
        pass

    print("\nEntering live prediction mode. Press Ctrl+C in this terminal to stop.\n")

    # -------------------------------------------------------------------------
    # 7.7 LIVE PREDICTION LOOP
    # -------------------------------------------------------------------------
    try:
        while True:
            # Pull latest sample from Physio stream (non-blocking).
            sample, ts = inlet.pull_sample(timeout=0.0)
            if sample and len(sample) > max(BVP_CH, EDA_CH):
                buf_bvp.append(sample[BVP_CH])
                buf_eda.append(sample[EDA_CH])

            # When enough data has accumulated for a window, process it.
            if len(buf_bvp) >= window_samples and len(buf_eda) >= window_samples:
                # Take the last window_samples to form current window.
                bvp_seg = np.array(buf_bvp[-window_samples:])
                eda_seg = np.array(buf_eda[-window_samples:])

                # Extract absolute features.
                feats_abs = extract_features(bvp_seg, eda_seg, FS)

                # Convert to relative features using baseline.
                feats_rel = rel_from_abs(feats_abs, baseline_means)

                # Choose which feature set to feed the model.
                if USE_RELATIVE_FEATURES_FOR_MODEL:
                    feat_names = [
                        "hr_mean_rel",
                        "hrv_rmssd_rel",
                        "eda_mean_rel",
                        "eda_std_rel",
                        "eda_deriv_std_rel"
                    ]
                    xvec = [[feats_rel[n] for n in feat_names]]
                else:
                    feat_names = [
                        "hr_mean",
                        "hrv_rmssd",
                        "eda_mean",
                        "eda_std",
                        "eda_deriv_std"
                    ]
                    xvec = [[feats_abs[n] for n in feat_names]]

                # Predict class label for this window.
                pred = model.predict(xvec)[0]

                # Try to obtain class probabilities (if the model supports predict_proba).
                try:
                    proba = model.predict_proba(xvec)[0]
                except Exception:
                    proba = None

                # Append the raw prediction to our smoothing buffer.
                pred_buf.append(pred)
                # smoothed = majority vote over last SMOOTH_K predictions.
                smoothed = Counter(pred_buf).most_common(1)[0][0]

                # Print diagnostic output to the console.
                if proba is not None:
                    prob_str = ", ".join(
                        f"{lbl}:{p:.2f}" for lbl, p in zip(LABELS, proba)
                    )
                    print(f"Pred: {pred} | Smoothed: {smoothed} | Proba: [{prob_str}]")
                else:
                    print(f"Pred: {pred} | Smoothed: {smoothed}")

                # Push results to LSL so other tools (Unreal, loggers, etc.) can read them.
                try:
                    outlet_state.push_sample([smoothed])
                    if proba is not None:
                        outlet_prob.push_sample(list(map(float, proba)))
                except Exception as e:
                    warn(f"LSL push error: {e}")

                # Slide our raw-signal windows forward.
                buf_bvp = buf_bvp[step_samples:]
                buf_eda = buf_eda[step_samples:]

            # Short sleep to avoid busy-waiting and reduce CPU load.
            time.sleep(0.01)

    except KeyboardInterrupt:
        # If the user presses Ctrl+C, we exit the loop gracefully.
        print("\nKeyboard interrupt received. Stopping predictions...")

    finally:
        # On exit, we send a final status="stopping" so any listeners can react.
        try:
            outlet_status.push_sample(["stopping"])
        except Exception:
            pass
        info("Real-time pipeline stopped cleanly.")


# =============================================================================
# 8. SCRIPT ENTRY
# =============================================================================

if __name__ == "__main__":
    main()
