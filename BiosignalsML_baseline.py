# =============================================================================
# BiosignalsML_baseline.py — Live predictions with baseline-aware features
# =============================================================================
#
# Overall pipeline:
#   1. Listen to an incoming LSL "Physio" stream (e.g., from BiosignalsLSL.py).
#      - Channels: BVP (blood volume pulse) + EDA (electrodermal activity).
#   2. Collect data into sliding windows (e.g. 5 seconds, overlapping).
#   3. For each window:
#        - Extract features from BVP (HR, HRV) and EDA (mean, std, derivative).
#        - Optionally accumulate windows labeled as "baseline" to estimate
#          per-user baseline levels.
#        - Create relative features: (current feature − baseline feature).
#        - Run a trained ML model to predict the current user state.
#   4. Output predictions via LSL:
#        - UserState:  string label (smoothed over last few predictions).
#        - UserStateProba: probability for each possible label.
#
# Baseline idea:
#   - Human physiology varies a lot between people (and across time).
#   - Baseline windows capture "normal" for a specific user (e.g., relaxed).
#   - Once we know their baseline, we can focus on *changes* from baseline,
#     which may generalize better than absolute values.
#
# Controls:
#   - Keyboard:
#       1,2,3,4 → assign manual label to current state (baseline/stress/relax/focus).
#       b       → toggle baseline-collection mode (treat all windows as baseline).
#       q       → quit.
#
#   - LSL Markers (optional):
#       If a "Markers" stream exists, its samples replace the keyboard label.
#
# =============================================================================

# -------------------------------
# 1. IMPORTS
# -------------------------------

# LSL objects:
# - StreamInlet: receive data from existing streams.
# - resolve_streams: discover all active LSL streams.
# - StreamInfo / StreamOutlet: create and publish our own output stream.
from pylsl import StreamInlet, resolve_streams, StreamInfo, StreamOutlet

# Numerical / data science stack:
import numpy as np          # Arrays and numerical operations
import pandas as pd         # Tabular data (e.g., logging features to CSV)

# Signal processing tools:
# - butter:   filter design (low-pass / band-pass).
# - filtfilt: zero-phase filtering (no lag).
# - find_peaks: peak detection for deriving HR/HRV from BVP.
from scipy.signal import butter, filtfilt, find_peaks

# Standard library goodies:
import threading            # For the keyboard label-input thread
import queue                # Thread-safe queue for communication (e.g., quit signal)
import time                 # Timing, periodic saving
import sys                  # Access to stdin for keypresses
import os                   # Saving CSV files to disk

from collections import deque, Counter  # Deque: fixed-size buffer, Counter: quick majority vote
import joblib               # Load pre-trained scikit-learn pipelines/models


# -------------------------------
# 2. CONFIGURATION CONSTANTS
# -------------------------------

# Indices of BVP and EDA channels in the Physio LSL stream.
# Note: these are zero-based: 0 = first channel.
BVP_CH, EDA_CH = 0, 1

# Sampling rate of the incoming Physio stream (in Hz).
FS = 10

# Window size (in seconds) we use to compute features.
WINDOW_SEC = 5

# Overlap between adjacent windows, expressed as a fraction.
# For example, 0.5 means 50% overlap.
OVERLAP = 0.5

# How often to save logged feature data to disk (in seconds).
SAVE_INTERVAL = 10

# EDA low-pass cutoff frequency (Hz) to smooth out rapid noise.
EDA_LP = 0.5

# Band-pass range (Hz) for BVP to isolate heart-related frequencies.
# Roughly 0.5–3 Hz ~ 30–180 bpm.
BVP_BAND = (0.5, 3.0)

# Path to the pre-trained model (scikit-learn pipeline or estimator).
MODEL_PATH = "Models/user_state_model.pkl"

# Whether the model expects *relative* features (i.e. *_rel) or *absolute* ones.
# If you retrain on *_rel features, set this to True.
USE_RELATIVE_FEATURES_FOR_MODEL = True

# How many labeled "baseline" windows we require before considering
# baseline means to be stable enough to use.
BASELINE_MIN_WINDOWS = 8

# Map numeric keys (as strings) to human-readable labels.
LABEL_MAP = {
    "1": "baseline",
    "2": "stress",
    "3": "relax",
    "4": "focus"
}


# -------------------------------
# 3. LOAD MODEL & PREPARE LSL OUTPUTS
# -------------------------------

# Load the serialized model. Usually this is a scikit-learn pipeline.
model = joblib.load(MODEL_PATH)

# Try to extract the class labels in the order the model uses.
# - If 'model' is a Pipeline with step named "clf", use that.
# - Otherwise, try using model.classes_ directly.
# - Fallback to a default label list.
try:
    LABELS = list(model.named_steps["clf"].classes_)
except Exception:
    try:
        LABELS = list(model.classes_)
    except Exception:
        LABELS = ["baseline", "stress", "relax"]

print("Model label order (probabilities will follow this order):", LABELS)

# --- LSL stream for the final predicted state (string label) ---
# Here we define a 1-channel stream of type "Classification".
info_state = StreamInfo(
    name="UserState",
    type="Classification",
    channel_count=1,
    nominal_srate=0,        # 0 = irregular (we push samples only when ready)
    channel_format="string",
    source_id="user_state"
)
outlet_state = StreamOutlet(info_state)

# --- LSL stream for the prediction probabilities (one channel per label) ---
info_prob = StreamInfo(
    name="UserStateProba",
    type="Probs",
    channel_count=len(LABELS),
    nominal_srate=0,       # again, event-based
    channel_format="float32",
    source_id="user_state_proba"
)

# Add channel descriptors for the probability stream: one label per channel.
prob_channels = info_prob.desc().append_child("channels")
for lbl in LABELS:
    ch = prob_channels.append_child("channel")
    ch.append_child_value("label", lbl)

outlet_prob = StreamOutlet(info_prob)

print("Broadcasting predictions over LSL:")
print("  - UserState       (string)")
print("  - UserStateProba  (float, one per class)")
print()


# -------------------------------
# 4. LABEL INPUT THREAD (KEYBOARD + BASELINE HOTKEY)
# -------------------------------

# This queue is used for sending a "quit" signal from the input thread
# back to the main loop.
label_queue = queue.Queue()

# Current label for the ongoing data (e.g. "baseline", "stress", etc.).
# Default: "unlabeled".
current_label = "unlabeled"

# If True, every window is treated as "baseline" for calibration,
# regardless of its label.
baseline_collect = False

def label_input_thread():
    """
    Background thread that listens for key presses from stdin.

    Keys:
        1,2,3,4 → change current_label according to LABEL_MAP.
        b       → toggle baseline collection mode ON/OFF.
        q       → send quit signal to main loop.
    """
    global current_label, baseline_collect

    print("\nKeyboard controls:")
    for k, v in LABEL_MAP.items():
        print(f"  {k} → {v}")
    print("  b → start/stop baseline calibration mode (collect baseline windows)")
    print("  q → quit script\n")

    # Read one character at a time from stdin.
    while True:
        key = sys.stdin.read(1)   # Blocking read: waits for a key press.
        if not key:
            continue

        k = key.lower()

        if k == "q":
            # Put a special message into the queue so the main loop can exit.
            label_queue.put("quit")
            break

        elif k == "b":
            # Flip the baseline collection flag.
            baseline_collect = not baseline_collect
            state = "ON" if baseline_collect else "OFF"
            print(f"[Baseline calibration mode]: {state}")

        elif key in LABEL_MAP:
            # Update the human-readable label for the current segment.
            current_label = LABEL_MAP[key]
            print(f"Current manual label: {current_label}")

# Start the label-input thread as a daemon so it dies when the main process ends.
threading.Thread(target=label_input_thread, daemon=True).start()


# -------------------------------
# 5. FILTER HELPER FUNCTIONS
# -------------------------------

def butter_bandpass(sig, fs, low, high, order=4):
    """
    Apply a band-pass Butterworth filter to 'sig'.

    Parameters:
        sig   : 1D sequence of signal values (e.g., raw BVP).
        fs    : sampling rate in Hz.
        low   : lower cutoff frequency in Hz.
        high  : upper cutoff frequency in Hz.
        order : filter order (higher → steeper).

    Returns:
        Filtered signal as a NumPy array. If the signal is too short to safely
        apply filtfilt, returns the original signal converted to float.
    """
    # Normalize cutoff frequencies to [0, 1] (0 = 0 Hz, 1 = Nyquist).
    b, a = butter(order, [low / (0.5 * fs), high / (0.5 * fs)], btype="band")

    # filtfilt needs a minimum number of samples (about 3×max(len(a), len(b))).
    if len(sig) > 3 * max(len(a), len(b)):
        return filtfilt(b, a, sig)
    else:
        # Not enough data, return a float array copy.
        return np.asarray(sig, float)


def butter_lowpass(sig, fs, cutoff, order=2):
    """
    Apply a low-pass Butterworth filter to 'sig' with safety checks.

    Parameters:
        sig    : 1D signal (e.g., raw EDA).
        fs     : sampling rate in Hz.
        cutoff : cutoff frequency in Hz.
        order  : filter order.

    Returns:
        Filtered signal if we have enough samples for filtfilt; otherwise,
        returns the unfiltered signal.
    """
    sig = np.asarray(sig, float)

    # Design the low-pass filter in normalized frequency units.
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")

    # Compute minimum pad length for filtfilt (how many points it needs).
    padlen = 3 * (max(len(a), len(b)) - 1)

    # If we don't have enough samples, skip filtering.
    if sig.size <= padlen:
        return sig

    # Apply filter in a try/except block to be robust against edge cases.
    try:
        return filtfilt(b, a, sig)
    except Exception:
        # If something goes wrong (e.g., numerical issues), just return original.
        return sig


def extract_features(bvp, eda, fs):
    """
    Extract window-level features from BVP and EDA signals.

    Returns a dict with:
        - hr_mean:      average heart rate (bpm) in the window.
        - hrv_rmssd:    RMSSD (root mean square of successive RR differences).
        - eda_mean:     mean EDA level.
        - eda_std:      standard deviation of EDA.
        - eda_deriv_std: std of first derivative of EDA (how "wobbly" it is).

    Steps:
        1. HR/HRV from BVP:
           - Clean BVP, detect peaks, convert peak distances to RR intervals.
        2. EDA stats:
           - Smooth EDA with a low-pass filter, compute mean/std and derivative std.
    """
    # --- HR/HRV from BVP peaks ---

    # Ensure we are working with a clean 1D float array.
    x = np.asarray(bvp, float).ravel()
    x = x[np.isfinite(x)]   # Remove any NaNs or infinities.

    # Initialize features as "no value yet".
    hr_mean = np.nan
    hrv_rmssd = np.nan

    # Require at least ~2 seconds of data to attempt HR.
    if x.size >= int(fs * 2):
        # Remove DC (center around zero).
        x = x - np.mean(x)
        x_std = np.std(x)

        if x_std > 1e-6:
            # Minimum distance between peaks in samples.
            # 0.5 seconds at fs = 10 Hz → distance = 5 samples.
            distance = int(max(1, 0.5 * fs))

            # Peak prominence scaled by signal std to make it adaptive.
            prominence = max(0.1 * x_std, 1e-3)

            try:
                # Detect peaks in the BVP signal.
                peaks, _ = find_peaks(x, distance=distance, prominence=prominence)

                # Need at least 3 peaks to reliably compute HRV.
                if peaks.size >= 3:
                    # RR intervals (in seconds).
                    rr = np.diff(peaks) / fs
                    rr = rr[rr > 0]   # discard non-positive intervals just in case.

                    if rr.size >= 2:
                        # Heart rate (bpm) for each interval = 60 / RR.
                        hr = 60.0 / rr
                        hr_mean = float(np.nanmean(hr))

                        # RMSSD = sqrt(mean((diff(RR))^2)).
                        hrv_rmssd = float(
                            np.sqrt(np.mean(np.square(np.diff(rr))))
                        )
            except Exception:
                # If peak detection fails, we simply keep NaNs for HR and HRV.
                pass

    # --- EDA smoothing + statistics ---

    # Smooth EDA with a low-pass filter.
    eda_s = butter_lowpass(eda, fs, cutoff=EDA_LP, order=2)

    # Convert to float array.
    eda_arr = np.asarray(eda_s, float).ravel()

    # Basic stats.
    eda_mean = float(np.mean(eda_arr)) if eda_arr.size else np.nan
    eda_std  = float(np.std(eda_arr))  if eda_arr.size else np.nan

    # Derivative statistics: how fast EDA is changing.
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


# -------------------------------
# 6. BASELINE ACCUMULATION
# -------------------------------

# List of dictionaries, each containing absolute features from one "baseline" window.
baseline_windows = []

# Flag that becomes True once we have enough baseline windows to be confident.
baseline_ready = False

# Pre-initialize baseline means for each feature we care about.
baseline_means = {
    k: np.nan for k in ["hr_mean", "hrv_rmssd", "eda_mean", "eda_std", "eda_deriv_std"]
}

def try_update_baseline(feats_abs, label_now):
    """
    Add the current window's absolute features to the baseline if:
      - baseline_collect is ON, OR
      - the current label is explicitly "baseline".

    Once enough baseline windows are collected (>= BASELINE_MIN_WINDOWS),
    compute the mean per feature (ignoring NaNs) and mark baseline_ready = True.
    """
    global baseline_ready, baseline_means

    # Decide whether this window should contribute to baseline.
    if baseline_collect or label_now == "baseline":
        # Keep only the feature keys we track in baseline_means.
        baseline_windows.append({
            k: float(feats_abs[k]) for k in baseline_means.keys()
        })

        # If we now have enough windows and baseline wasn't ready before, compute it.
        if len(baseline_windows) >= BASELINE_MIN_WINDOWS and not baseline_ready:
            # Use a DataFrame for convenience; nanmean ignores missing values.
            dfb = pd.DataFrame(baseline_windows)
            baseline_means = {
                k: float(np.nanmean(dfb[k].values)) for k in baseline_means.keys()
            }
            baseline_ready = True
            print(f"[Baseline computed from {len(baseline_windows)} windows]")
            print("  Means:", baseline_means)

def make_relative(feats_abs):
    """
    Build a dict of relative features: (feature - baseline_mean).

    If baseline is not ready yet, we return zeros for all *_rel features
    to avoid contaminating the model with arbitrary offsets.
    """
    rel = {}

    for k, v in feats_abs.items():
        b = baseline_means.get(k, np.nan)
        if baseline_ready and np.isfinite(b):
            rel[k + "_rel"] = float(v) - float(b)
        else:
            # Before baseline is ready, all relative features are 0.
            rel[k + "_rel"] = 0.0

    return rel


# -------------------------------
# 7. SLIDING WINDOW BUFFERS
# -------------------------------

# Size of one feature-extraction window in samples.
window_samples = int(WINDOW_SEC * FS)

# Step size between windows (how far we slide each time) in samples.
# With OVERLAP = 0.5, we move half a window each time.
step_samples = int(max(1, window_samples * (1 - OVERLAP)))

# Buffers to hold raw BVP and EDA data until we have enough for a window.
buf_bvp, buf_eda = [], []

# For optional logging of feature rows; currently not filled in this script but
# kept for extension (e.g., appending dicts of features + labels over time).
data_log = []

# Timestamp of last periodic save (for the logging feature).
last_save = time.time()


# -------------------------------
# 8. CONNECT TO LSL PHYSIO & MARKER STREAMS
# -------------------------------

print("\nSearching for available LSL streams...")
streams = resolve_streams()

physio_stream = None   # Will store the LSL stream that carries BVP+EDA.
marker_stream = None   # Optional: LSL stream carrying external labels/markers.

# Iterate over all discovered streams and pick the ones we care about by type.
for s in streams:
    if s.type() == "Physio":
        physio_stream = s
    elif s.type() == "Markers":
        marker_stream = s

# We require a Physio stream to proceed.
if not physio_stream:
    raise SystemExit("No Physio stream found. Start BiosignalsLSL.py / OpenSignals bridge.")

# Create an inlet for the Physio stream.
inlet = StreamInlet(physio_stream)
print(f"Connected to Physio stream: {inlet.info().name()} ({inlet.info().type()})")

# Marker inlet is optional: if available, it can override keyboard labels.
marker_inlet = None
if marker_stream:
    marker_inlet = StreamInlet(marker_stream)
    print(f"Connected to Marker stream: {marker_inlet.info().name()} ({marker_inlet.info().type()})")
else:
    print("No Marker stream found — using manual labeling / baseline mode only.")

print("\nStarting live predictions (baseline-aware)...")
print(f"Baseline mode: press 'b' to toggle collection. Required windows: {BASELINE_MIN_WINDOWS}\n")


# -------------------------------
# 9. MAIN LIVE PROCESSING LOOP
# -------------------------------

try:
    while True:
        # 9.1 Read a sample from the Physio stream (non-blocking).
        sample, ts = inlet.pull_sample(timeout=0.0)

        # Make sure we have a valid sample and enough channels for BVP and EDA.
        if sample and len(sample) > max(BVP_CH, EDA_CH):
            buf_bvp.append(sample[BVP_CH])
            buf_eda.append(sample[EDA_CH])

        # 9.2 If a Marker stream is available, check if a new marker arrived.
        if marker_inlet:
            marker, mts = marker_inlet.pull_sample(timeout=0.0)
            if marker:
                # Overwrite current_label with the marker value (e.g., "baseline", "stress", etc.).
                current_label = marker[0]

        # 9.3 Check for quit signal from the keyboard thread.
        if not label_queue.empty() and label_queue.get_nowait() == "quit":
            print("Quit signal received from keyboard thread.")
            break

        # 9.4 If we have enough samples in the buffers, process a window.
        if len(buf_bvp) >= window_samples and len(buf_eda) >= window_samples:
            # Take the last 'window_samples' data points as the current window.
            bvp_seg = np.array(buf_bvp[-window_samples:])
            eda_seg = np.array(buf_eda[-window_samples:])

            # Extract absolute features from this window.
            feats_abs = extract_features(bvp_seg, eda_seg, FS)

            # Possibly update baseline (if in baseline_collect mode or labeled "baseline").
            try_update_baseline(feats_abs, current_label)

            # Build the feature vector for the model.
            # Either use relative features (default here) or absolute features.
            feats_rel = make_relative(feats_abs)

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

            # 9.5 Predict user state with the model.
            predicted_state = model.predict(xvec)[0]

            # Try to also get class probabilities.
            try:
                proba = model.predict_proba(xvec)[0]
            except Exception:
                proba = None

            # 9.6 Temporal smoothing of predictions.
            # We keep the last few predicted labels and return the majority vote.
            if "pred_smooth_buf" not in globals():
                pred_smooth_buf = deque(maxlen=3)
            pred_smooth_buf.append(predicted_state)
            smoothed = Counter(pred_smooth_buf).most_common(1)[0][0]

            # 9.7 Print a status line, including baseline readiness.
            if baseline_ready:
                baseline_tag = "READY"
            else:
                remaining = max(0, BASELINE_MIN_WINDOWS - len(baseline_windows))
                baseline_tag = f"collect {remaining}"

            if proba is not None:
                # Format probabilities as "label:0.12, label:0.34, ..."
                prob_str = ", ".join(
                    [f"{lbl}:{p:.2f}" for lbl, p in zip(LABELS, proba)]
                )
                print(
                    f"[Baseline:{baseline_tag}] "
                    f"Pred: {predicted_state} | Smoothed: {smoothed} | "
                    f"Proba: [{prob_str}]"
                )
            else:
                print(
                    f"[Baseline:{baseline_tag}] "
                    f"Pred: {predicted_state} | Smoothed: {smoothed}"
                )

            # 9.8 Push predictions to the LSL outputs.
            try:
                # State is a single string channel.
                outlet_state.push_sample([smoothed])

                # Probabilities: one float per label, if available.
                if proba is not None:
                    outlet_prob.push_sample(list(map(float, proba)))
            except Exception as e:
                print("LSL push error:", e)

            # 9.9 Slide the window forward by 'step_samples':
            # Drop the oldest samples from the buffers.
            buf_bvp = buf_bvp[step_samples:]
            buf_eda = buf_eda[step_samples:]

        # 9.10 Periodic save of logged features (if any).
        # (Note: in this version, data_log is never filled, but the structure is ready.)
        if time.time() - last_save > SAVE_INTERVAL and data_log:
            try:
                os.makedirs("Data", exist_ok=True)
                df = pd.DataFrame(data_log)
                df.to_csv("Data/live_features_log.csv", index=False)
                print(f"Saved {len(df)} feature rows to Data/live_features_log.csv")
                last_save = time.time()
            except Exception as e:
                print("Save error:", e)

except KeyboardInterrupt:
    # Handle Ctrl+C gracefully.
    print("\nStopped by user (Ctrl+C)")

finally:
    # -------------------------------
    # 10. FINAL SAVE ON EXIT
    # -------------------------------
    try:
        if data_log:
            os.makedirs("Data", exist_ok=True)
            df = pd.DataFrame(data_log)
            df.to_csv("Data/live_features_log_final.csv", index=False)
            print(f"\nFinal save: {len(df)} rows written to Data/live_features_log_final.csv")
        else:
            print("No data collected — final CSV not created.")
    except Exception as e:
        print("Final save error:", e)
