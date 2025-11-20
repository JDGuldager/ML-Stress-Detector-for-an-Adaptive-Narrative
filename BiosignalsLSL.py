# =========================
# 1. IMPORTS
# =========================

# pylsl is the Python interface to the Lab Streaming Layer (LSL).
# - StreamInlet:   lets us RECEIVE data from an existing LSL stream.
# - StreamOutlet:  lets us SEND (publish) our own LSL stream.
# - StreamInfo:    describes the stream we create (name, channels, etc.).
# - resolve_streams: searches the network for active LSL streams.
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_streams  

# collections: gives us deque, which is like a list with a maximum length (perfect for signal buffers).
# threading:  lets us run the data-reading loop in the background.
# time:       used for timing (sleep) and dealing with timestamps.
import collections, threading, time  

# NumPy: the standard numerical library for arrays, mean, std, etc.
import numpy as np  

# SciPy’s signal module:
# - butter:   designs a digital Butterworth filter.
# - filtfilt: applies the filter forward and backward (zero-phase) to avoid delays/distortion.
from scipy.signal import butter, filtfilt  


# =========================
# 2. CONFIGURATION CONSTANTS
# =========================

# Channel indices in the *incoming* PLUX stream for BVP and EDA.
# Note: these are zero-based indices: 0 = first channel, 1 = second, etc.
BVP_CH, EDA_CH = 1, 2             

# How often we want to process / send data onward (in samples per second).
SAMPLING_RATE = 10                

# Maximum number of recent samples we keep in memory for each signal.
# Once this limit is reached, old samples fall out automatically.
BUFFER_SIZE = 200                 

# Cutoff frequency (Hz) for low-pass filtering the EDA signal.
# This is because EDA is a slow-changing signal; high-frequency noise is not interesting.
EDA_LP_CUTOFF = 1.0              

# Minimum allowed time (in seconds) between two BVP peaks.
# This prevents detecting multiple "peaks" within a single heartbeat due to noise.
MIN_PEAK_INTERVAL = 0.5          

# We don’t want to run a filter on only a handful of samples,
# so we wait until we have at least this many before low-pass filtering EDA.
EDA_MIN_SAMPLES = 30             


# =========================
# 3. LSL INPUT: CONNECT TO PLUX
# =========================

print("Searching for PLUX LSL stream...")

# Ask LSL to give us a list of all currently visible streams.
# If OpenSignals (or similar software) is streaming, PLUX should appear here.
streams = resolve_streams()  

# If no stream is found, there is nothing to read from → exit the program with a clear message.
if not streams:
    raise SystemExit("No LSL input stream found (start OpenSignals).")

# Create an "inlet" (receiver) for the first discovered stream.
# In many setups, you only have one PLUX stream, so this is fine.
inlet = StreamInlet(streams[0])  

# Print some basic info about the connected stream so the user knows it worked.
print(f"Connected to LSL stream: {inlet.info().name()} "
      f"({inlet.info().channel_count()} channels)")


# =========================
# 4. LSL OUTPUT: STREAM TOWARD UNREAL ENGINE
# =========================

# We now define our own outgoing LSL stream that Unreal Engine will read from.
# StreamInfo essentially describes what our stream looks like.
info = StreamInfo(
    name="BioBridge",          # A human-readable name for the stream.
    type="Physio",             # A category/type label (Unreal can filter by this).
    channel_count=3,           # We will send 3 channels: BVP, EDA, and HR.
    nominal_srate=SAMPLING_RATE,  # Sampling rate we intend to output at (10 Hz).
    channel_format="float32",  # Data type of each sample.
    source_id="plux_biobridge" # A unique identifier for this stream instance.
)

# Add more detailed meta-information: the labels for each channel.
# This is helpful for clients (like Unreal) to know what each channel represents.
channels = info.desc().append_child("channels")
for label in ["BVP", "EDA", "HR"]:
    ch = channels.append_child("channel")
    ch.append_child_value("label", label)

# Create the actual outlet (publisher) using the StreamInfo we just defined.
outlet = StreamOutlet(info)

print("Broadcasting to Unreal via LSL:")
print("  StreamName = 'BioBridge'")
print("  StreamType = 'Physio'")
print()


# =========================
# 5. BUFFERS AND GLOBAL STATE
# =========================

# We will keep the most recent samples in these buffers.
# deque with maxlen behaves like a sliding window: it automatically discards the oldest entries.
bvp_buf = collections.deque(maxlen=BUFFER_SIZE)  # BVP signal values
eda_buf = collections.deque(maxlen=BUFFER_SIZE)  # EDA signal values
ts_buf  = collections.deque(maxlen=BUFFER_SIZE)  # Timestamps corresponding to the samples

# For heart rate, we don’t need all past peaks, only the last few.
# We’ll keep the timestamps of the last 5 detected BVP peaks.
last_peaks = collections.deque(maxlen=5)

# A lock is needed because both the background reader thread and the main loop
# will access these buffers. The lock prevents them from modifying the buffers at the same time.
lock = threading.Lock()

# This boolean flag controls the background reader thread:
# when set to False, the reader will stop running.
running = True


# =========================
# 6. HELPER FUNCTIONS
# =========================

def detect_peak(buf):
    """
    Decide whether the latest BVP samples contain a "peak" (a heartbeat).

    How it works (simple heuristic):
    - Require at least 3 samples in the buffer (otherwise you can’t have a "middle" point).
    - Look at the last 3 samples: (a, b, c).
    - A peak is detected if:
        * b is larger than both a and c (local maximum), AND
        * b is also significantly higher than the overall level:
              b > mean(buf) + 0.3 * std(buf)

    This second condition is a crude way to avoid counting tiny noise bumps as peaks.
    """

    # If the buffer is too short, we simply cannot detect a peak yet.
    if len(buf) < 3:
        return False

    # Take the last three samples from the buffer.
    a, b, c = buf[-3:]  

    # Compute a dynamic threshold based on the buffer's mean and standard deviation.
    # "0.3" is a tuning parameter: higher → fewer peaks (more strict), lower → more peaks.
    threshold = np.mean(buf) + 0.3 * np.std(buf)

    # Return True if:
    # - b is a local maximum (higher than neighbors) AND
    # - b is above the threshold (likely a true heartbeat)
    return (b > a) and (b > c) and (b > threshold)


def lowpass_safe(sig, cutoff=EDA_LP_CUTOFF, fs=SAMPLING_RATE, order=4):
    """
    Safely apply a low-pass Butterworth filter to the EDA signal.

    Why "safe"? Because filtfilt (zero-phase filtering) needs a minimum number of samples:
    with too few samples, it can misbehave or raise errors. So we:
      1) Check that we have enough data.
      2) If not, just return the original signal unchanged.
      3) If we do, design a filter and apply filtfilt.

    Parameters:
        sig    : 1D sequence of EDA samples
        cutoff : cutoff frequency in Hz
        fs     : sampling rate in Hz
        order  : filter order (higher = steeper, but also more "demanding")
    """
    # Ensure we are working with a NumPy array of floats (filtfilt expects this).
    sig = np.asarray(sig, dtype=float)

    # filtfilt needs at least (3 * order + 1) samples to run safely.
    # We also require at least EDA_MIN_SAMPLES for meaningful smoothing.
    min_len = max(EDA_MIN_SAMPLES, 3 * order + 1)

    if len(sig) < min_len:
        # Not enough data yet: just return the unfiltered signal.
        # The caller will still get a usable value, just not smoothed.
        return sig

    # Design a Butterworth low-pass filter.
    # Nyquist frequency is half the sampling rate.
    nyq = 0.5 * fs
    normalized_cutoff = cutoff / nyq  # between 0 and 1

    # Get the filter coefficients (b, a) for the chosen order and cutoff.
    b, a = butter(order, normalized_cutoff, btype="low")

    # filtfilt applies the filter forward and backward in time,
    # which removes phase shifts (no lag in the output).
    filtered = filtfilt(b, a, sig)

    return filtered


# =========================
# 7. BACKGROUND READER THREAD
# =========================

def reader():
    """
    Background function that continuously pulls samples from the PLUX LSL stream.

    This runs in its own thread so the main thread can focus on processing and
    sending data out to Unreal.

    What it does, in a loop:
    - Pull one sample from the inlet.
    - Extract BVP and EDA channels.
    - Append them (and the timestamp) to the corresponding buffers.
    - Sleep a bit to approximate the target sampling rate.
    """
    while running:
        # Try to pull a sample from the LSL stream.
        # timeout=0.0 means: don't wait if there is no new sample.
        sample, ts = inlet.pull_sample(timeout=0.0)

        # If we either got no sample, or the sample doesn't have enough channels,
        # we skip this iteration and try again shortly.
        if not sample or len(sample) <= max(BVP_CH, EDA_CH):
            time.sleep(0.05)  # brief pause before next attempt
            continue

        # We are going to modify the shared buffers, so we acquire the lock.
        with lock:
            # Convert to float just to be safe (sometimes LSL samples may not be pure floats).
            bvp_buf.append(float(sample[BVP_CH]))
            eda_buf.append(float(sample[EDA_CH]))
            ts_buf.append(ts)

        # Sleep to roughly match the desired processing rate.
        # (In a real system, you might want to carefully align this with the actual device rate.)
        time.sleep(1.0 / SAMPLING_RATE)


# Start the reader in a separate, "daemon" thread.
# Daemon=True means the thread will automatically stop when the main program exits.
threading.Thread(target=reader, daemon=True).start()


# =========================
# 8. MAIN PROCESSING LOOP
# =========================

print("Streaming BVP, EDA, and HR to Unreal at ~10 Hz.")
print("Press Ctrl+C in this window to stop.\n")

try:
    while True:
        # This sleep defines how often the main loop executes
        # (and thus how often we push data to Unreal).
        time.sleep(1.0 / SAMPLING_RATE)

        # Copy the buffers under the lock to avoid them being modified
        # while we are reading from them.
        with lock:
            # If we don't have any data yet, we can't do anything; skip this cycle.
            if not bvp_buf:
                continue

            # Make local copies of the current buffer contents.
            bvp_vals = list(bvp_buf)
            eda_vals = list(eda_buf)
            ts_vals  = list(ts_buf)

        # -------------------------
        # 8.1 HEART RATE ESTIMATION
        # -------------------------

        # Initialize HR as "not a number".
        # This is a common way to say "we don't have a valid value yet".
        hr_bpm = np.nan  

        # Check if the latest BVP samples contain a newly detected peak.
        if detect_peak(bvp_vals):
            # Use the timestamp of the most recent sample as the peak time.
            t = ts_vals[-1]

            # Only accept this as a new peak if:
            # - we have no previous peaks (first heartbeat), OR
            # - enough time has passed since the last detected peak.
            if not last_peaks or (t - last_peaks[-1]) > MIN_PEAK_INTERVAL:
                last_peaks.append(t)

                # If we have at least two peaks, we can compute intervals between them.
                if len(last_peaks) > 1:
                    # np.diff computes differences between consecutive elements.
                    rr_intervals = np.diff(last_peaks)  # in seconds

                    # Sanity check: all intervals should be positive.
                    if rr_intervals.size > 0 and np.all(rr_intervals > 0):
                        # Heart rate = 60 / mean RR (RR is time between beats in seconds).
                        hr_bpm = 60.0 / np.mean(rr_intervals)

        # -------------------------
        # 8.2 EDA FILTERING
        # -------------------------

        # Smooth the entire EDA buffer. If there aren't enough samples yet,
        # lowpass_safe will just return the raw values.
        eda_smoothed = lowpass_safe(eda_vals)

        # Take the most recent EDA value from the smoothed signal.
        # (The index -1 means "last element"). As a fallback, if for some reason
        # the smoothed array is empty, we use the last raw EDA value.
        eda_val = float(eda_smoothed[-1]) if len(eda_smoothed) > 0 else float(eda_vals[-1])

        # -------------------------
        # 8.3 SEND SAMPLE TO UNREAL
        # -------------------------

        # Prepare one "sample" containing:
        #   - latest BVP value
        #   - latest (smoothed) EDA value
        #   - current HR estimate (or NaN if unknown)
        sample_out = [float(bvp_vals[-1]), eda_val, float(hr_bpm)]

        # Push the sample into our outgoing LSL stream.
        # Any LSL client (like an Unreal plugin) connected to "BioBridge" will receive it.
        outlet.push_sample(sample_out)

        # -------------------------
        # 8.4 OPTIONAL CONSOLE DEBUG
        # -------------------------

        # Print a small debug line so you can see what the script is doing in real time.
        # Timestamps and signals are formatted with a fixed number of decimal places for readability.
        print(
            f"{ts_vals[-1]:.3f} | "
            f"BVP: {bvp_vals[-1]:+8.4f} | "
            f"EDA: {eda_val:+8.4f} | "
            f"HR: {hr_bpm:6.1f}"
        )

except KeyboardInterrupt:
    # This block is executed when you press Ctrl+C.
    # We set the running flag to False so the background reader thread
    # can stop its loop cleanly.
    running = False
    print("\nKeyboard interrupt received. Stopping streaming cleanly...")
