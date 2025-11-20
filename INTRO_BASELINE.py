from psychopy import visual, core, event
from pylsl import StreamInlet, resolve_streams

# --- Try to connect to LSL status + baseline progress streams (non-blocking) ---

status_inlet = None
progress_inlet = None
use_lsl = False

print("Trying to find LSL streams 'UserStateStatus' and 'UserStateBaselineProgress'...")

try:
    # Try for up to ~2 seconds (if this pylsl version supports timeout)
    streams = resolve_streams(2.0)
except TypeError:
    # Older pylsl: no timeout argument
    streams = resolve_streams()

for s in streams:
    if s.name() == "UserStateStatus":
        status_inlet = StreamInlet(s)
        print("Connected to UserStateStatus")
    elif s.name() == "UserStateBaselineProgress":
        progress_inlet = StreamInlet(s)
        print("Connected to UserStateBaselineProgress")

if status_inlet is not None and progress_inlet is not None:
    use_lsl = True
    print("Using LSL for status + progress.")
else:
    print("WARNING: Could not find both LSL streams, falling back to local timer baseline.")
    use_lsl = False

# --- PsychoPy Window ---
win = visual.Window(fullscr=True, color='black', units='height')

# --- Neutral baseline with shapes + progress bar ---

message = visual.TextStim(
    win,
    text=(
        "Baseline recording.\n\n"
        "Please look at the shapes, remain seated, and avoid movement.\n"
        "Breathe naturally.\n"
    ),
    pos=(0, -0.35),
    color='white',
    height=0.025,
    wrapWidth=1.2
)

# Geometric shapes
shapes = [
    visual.ShapeStim(win, vertices='cross', size=0.1, lineColor='white', fillColor=None),
    visual.Circle(win, radius=0.05, lineColor='white', fillColor=None),
    visual.Rect(win, width=0.1, height=0.1, lineColor='white', fillColor=None),
    visual.Polygon(win, edges=3, radius=0.06, lineColor='white', fillColor=None),
    visual.Polygon(win, edges=4, radius=0.06, lineColor='white', fillColor=None),
    visual.Polygon(win, edges=5, radius=0.06, lineColor='white', fillColor=None),
    visual.Polygon(win, edges=6, radius=0.06, lineColor='white', fillColor=None),
    visual.Circle(win, radius=0.06, lineColor='white', fillColor='white'),
    visual.Polygon(win, edges=8, radius=0.06, lineColor='white', fillColor=None),
    visual.Polygon(win, edges=7, radius=0.06, lineColor='white', fillColor=None),
    visual.Polygon(win, edges=9, radius=0.06, lineColor='white', fillColor=None),
    visual.Polygon(win, edges=10, radius=0.06, lineColor='white', fillColor=None)
]

# Progress bar visuals
BAR_WIDTH = 0.8
BAR_HEIGHT = 0.03
BAR_Y = -0.45

bar_bg = visual.Rect(
    win,
    width=BAR_WIDTH,
    height=BAR_HEIGHT,
    pos=(0, BAR_Y),
    lineColor='white',
    fillColor=None
)

# Foreground bar starts at width 0 and we’ll update its width + pos each frame.
# No anchorHoriz here; we’ll simulate "left-anchored" by shifting the center.
bar_fg = visual.Rect(
    win,
    width=0.0,
    height=BAR_HEIGHT * 0.8,
    pos=(-BAR_WIDTH / 2.0, BAR_Y),  # temporary initial position
    lineColor=None,
    fillColor='white'
)

# Fallback baseline duration if LSL is not available
BASELINE_DURATION_FALLBACK = 120.0  # seconds
baseline_timer = core.Clock()
shape_timer = core.Clock()
shape_index = 0

current_status = "calibrating"
progress_val = 0.0
baseline_done = False

while not baseline_done:
    # Early abort
    if 'escape' in event.getKeys(keyList=['escape']):
        win.close()
        core.quit()

    # --- LSL: update status and progress, if available ---
    if use_lsl:
        if status_inlet is not None:
            s_sample, _ = status_inlet.pull_sample(timeout=0.0)
            if s_sample:
                current_status = s_sample[0]

        if progress_inlet is not None:
            p_sample, _ = progress_inlet.pull_sample(timeout=0.0)
            if p_sample:
                progress_val = float(p_sample[0])
                if progress_val < 0.0:
                    progress_val = 0.0
                elif progress_val > 1.0:
                    progress_val = 1.0

        # Baseline ends when RunRealtime sends "ready"
        if current_status == "ready":
            baseline_done = True

    else:
        # Fallback: local timer-based "fake" progress
        elapsed = baseline_timer.getTime()
        progress_val = min(1.0, elapsed / BASELINE_DURATION_FALLBACK)
        if elapsed >= BASELINE_DURATION_FALLBACK:
            baseline_done = True

    # --- Update shapes every 6 seconds ---
    if shape_timer.getTime() >= 6.0:
        shape_index = (shape_index + 1) % len(shapes)
        shape_timer.reset()

    # --- Draw shapes + message ---
    shapes[shape_index].draw()
    message.draw()

    # --- Draw progress bar ---
    bar_bg.draw()
    # make the bar grow from the left: width = BAR_WIDTH * progress, center shifts accordingly
    bar_fg.width = BAR_WIDTH * progress_val
    bar_fg.pos = (-BAR_WIDTH / 2.0 + (bar_fg.width / 2.0), BAR_Y)
    bar_fg.draw()

    win.flip()

# --- End screen ---
end_text = visual.TextStim(
    win,
    text="Baseline recording is complete.\n\nPress SPACE to close this window and continue.",
    color='white',
    height=0.03
)
end_text.draw()
win.flip()
event.waitKeys(keyList=['space', 'escape'])

win.close()
core.quit()
