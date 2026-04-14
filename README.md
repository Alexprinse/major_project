## Troubleshooting

**Camera View Not Changing Responsively:**
- Camera view switching operates independently of lane detection
- If TAB/Y keys feel unresponsive, the lane detection computation may be blocking the event loop
- Press `L` to disable lane detection overlay (improves responsiveness immediately)
- View switching still works with lane detection on, but may have slight lag
- Full dashboard restart: Press `ESC` to exit, then rerun the app

**Red Line Detection False Positives:**
- False positives occur when non-red UI elements or scene objects trigger the HSV red mask
- Use the debug visualization tool (shown above) to see preprocessing steps in real-time
- Adjust HSV thresholds in `src/drowsiness_detector/lane_detection.py` > `_detect_red_borders()` method:
	- Increase saturation lower bound (50 → 80 or higher) to skip dull colors
	- Adjust Hough line detection parameters (`threshold`, `minLineLength`, `maxLineGap`)
	- Stricter thresholds may reduce false positives but might miss real red lines
- Temporarily disable with `L` key if needed for specific driving scenes
- Red line detection is optional; you can rely solely on white lane detection by keeping lane mode on but ignoring red detection output

## Test the detector
# Drowsiness Detection + CARLA Safety Controller

This workspace is a starter scaffold for a drowsiness detection system that can:

- detect drowsiness from video or sensor-derived signals,
- trigger an audible alert when the driver appears sleepy,
- request a safe pull-over / park maneuver in CARLA if drowsiness continues.

The current codebase is a clean starting point with optional dependencies and placeholder control flow so it can be extended with a real computer-vision model or a CARLA vehicle controller.

## Project Layout

- `src/drowsiness_detector/detector.py` - drowsiness scoring and state tracking
- `src/drowsiness_detector/alerts.py` - beep / alert handling
- `src/drowsiness_detector/carla_controller.py` - CARLA integration hooks
- `src/drowsiness_detector/app.py` - example orchestration entry point
- `requirements.txt` - runtime dependencies

## Setup

1. Create a Python environment.
2. Install the package in editable mode:

```bash
python3 -m pip install -e .
```

If you prefer the flat requirements file, you can also install from:

```bash
python3 -m pip install -r requirements.txt
```

## Run the demo

The current entry point is a safe simulation scaffold that you can extend with your model and CARLA bridge:

```bash
PYTHONPATH=src python3 -m drowsiness_detector.app --demo
```

## Run with CARLA

Start CARLA first, make sure your ego vehicle has a role name such as `hero`, then run:

```bash
PYTHONPATH=src python3 -m drowsiness_detector.app --camera --carla --carla-host 127.0.0.1 --carla-port 2000 --carla-role-name hero
```

If CARLA has no ego vehicle yet, spawn one automatically and open both windows (CARLA driver view + webcam drowsiness view):

```bash
PYTHONPATH=src python3 -m drowsiness_detector.app --camera --carla --spawn-vehicle-if-missing --carla-role-name hero
```

To enable lane detection overlay on the CARLA camera view:

```bash
PYTHONPATH=src python3 -m drowsiness_detector.app --camera --carla --spawn-vehicle-if-missing --carla-role-name hero --lane-detection
```

To start directly with Cosmos control visualization camera mode (if available in your CARLA build):

```bash
PYTHONPATH=src python3 -m drowsiness_detector.app --camera --carla --spawn-vehicle-if-missing --carla-role-name hero --carla-camera-mode cosmos --lane-detection
```

**Note:** Cosmos visualization includes **red border detection** (detects footpath/curb lines visible in Cosmos rendering). The dashboard displays detected red border position (LEFT, CENTER, RIGHT) in telemetry.

To run parking tuning mode with Cosmos visualization and lane detection:

```bash
PYTHONPATH=src python3 -m drowsiness_detector.app --parking-tuning --spawn-vehicle-if-missing --carla-role-name hero
```

In parking tuning mode:
- Cosmos visualization displays control/debug data overlaid on CARLA view
- Lane detection is always enabled and shows steering hints
- Manual controls let you test steering/throttle before running autonomous parking
- **SPACE** to trigger autonomous roadside parking
- Telemetry panel shows parking status, steps executed, speed, lane offset, steering input

Dashboard layout in this mode:

- Center/left: CARLA driver view (primary tab, starts at Front view)
- Top-right: webcam eye/drowsiness feed
- Right panels: speed monitor, vehicle mode, driver state, transmission, weather, slowdown status, camera label
- The active CARLA camera view name is shown in the header and on the road-view panel.
- **Tabbed camera system**: Press `TAB` to cycle through camera views (primary tab with Front view as default), or press `Y` to jump to HIGH CHASE view (secondary tab for high elevation rear-view).
- Press `F11` to toggle fullscreen and return to windowed mode. The dashboard will reflow to the current window size.
- Use the mouse wheel over the telemetry panel to scroll detailed vehicle and safety information.
- `TAB` now cycles through more than three camera views, including front-left, front-right, rear-left, rear-right, and high chase.

Keyboard controls in the CARLA window:

- `W`: throttle
- `S`: brake
- `A` / `D`: steer
- `SPACE`: handbrake
- `Q` or `R`: toggle reverse
- `P`: toggle CARLA autopilot
- `TAB`: next camera view (cycles through all 8 views, primary tab)
- `SHIFT+TAB`: previous camera view
- `1` to `8`: jump directly to a camera view
- `Y`: switch to HIGH CHASE view (secondary tab)
- `V`: toggle camera mode between RGB and Cosmos visualization
- If `sensor.camera.cosmos_visualization` is unavailable, mode falls back to RGB automatically
- `C`: cycle weather
- `SHIFT+C`: cycle weather backwards
- `M`: toggle manual transmission
- `L`: toggle lane detection overlay on CARLA view
- `,` / `.`: gear down / gear up (manual transmission mode)
- `ESC`: quit dashboard

Drowsiness safety behavior in CARLA mode:

- Short blinks are filtered: low-eye signal must persist for about 2.5 seconds before `DROWSY`.
- Driver state `DROWSY` or `SLEEPY` automatically applies gentle deceleration.
- If `SLEEPY` persists to pull-over threshold, the system attempts autonomous roadside parking using CARLA routing.
- Vehicle speed is shown live in both the CARLA window and webcam overlay.

Parking notes:

- Full roadside routing uses CARLA `BasicAgent` (requires `shapely`).
- If agent dependencies are unavailable, the system uses a fallback right-biased deceleration maneuver and then stops.

When the detector reaches the pull-over threshold, it will apply a hard stop to the attached CARLA vehicle. This is the first integration step; you can extend it later to steer into a shoulder or parking bay.

## Standalone Lane Detection

Run the lane detection visualization by itself before integrating it into the CARLA dashboard:

```bash
PYTHONPATH=src python3 -m drowsiness_detector.lane_detection --source 0
```

Use a video file instead of a webcam by passing its path to `--source`. The lane demo shows lane lines, lane center, vehicle center, and steering hints on the video frame.

Lane detection also includes **red border/footpath detection** (Cosmos visualization) which:
- Detects red lines (footpath/curb boundaries) in the Cosmos control visualization
- Automatically uses detected red lines as lane left/right boundaries
- Calculates lane width and verifies it's larger than the vehicle width
- Shows lane center and vehicle offset for parking guidance
- Displays detection status in dashboard telemetry

## Debug Red Line Detection

If red line detection is not working or giving false positives, use the debug visualization tool to see all preprocessing steps:

```bash
# Visualize with webcam
PYTHONPATH=src python3 -m drowsiness_detector.red_line_debug --camera-index 0

# Visualize with CARLA Cosmos camera
PYTHONPATH=src python3 -m drowsiness_detector.red_line_debug --carla --spawn-vehicle-if-missing --carla-role-name hero
```

This displays a 2x2 grid:
- **Top-left**: Original frame
- **Top-right**: Red mask (HSV color space filtering)
- **Bottom-left**: After morphological operations (cleanup)
- **Bottom-right**: Canny edges (what gets sent to line detection)

**Troubleshooting False Detection:**

**Disabling Red Line Detection Temporarily:**
**Recent Improvements to Reduce False Positives:**
- The detection system was improved with stricter thresholds to minimize false detections:
	- **HSV saturation** increased from 50 to 80 (now skips dull/washed-out colors)
	- **Red pixel ratio check**: scenes with <1% red content are filtered out immediately
	- **Hough threshold** increased from 30 to 50 (requires stronger edge clusters)
	- **Minimum line length** increased from 50 to 80 pixels (longer boundaries only)
	- **Vertical line filtering**: lines must be lane-like (slope > 0.5), not horizontal UI elements
	- **Position validation**: confirmed left line is left of right line (prevents invalid pairs)
- These changes should significantly reduce false positives in Cosmos visualization

**Troubleshooting False Detection:**
- If red mask shows unwanted colors, adjust HSV saturation ranges (increase lower bound to ~100)
- If Canny edges are too noisy, increase Canny thresholds
- If lines aren't detected, you need longer continuous red segments in your scene
- Press `L` in dashboard to toggle lane detection off if it's interfering

**Disabling Red Line Detection Temporarily:**
- The debug tool lets you assess whether to enable/disable in your scene
- Red detection can be toggled by temporarily disabling lane detection (`L` key in dashboard)
- Full red line thresholds are in `src/drowsiness_detector/lane_detection.py` > `_detect_red_borders()`

## Test the detector

Run the unit tests that cover the drowsiness state machine and pull-over timing:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Next implementation steps

1. Plug in a real face/eye detection pipeline.
2. Calibrate the drowsiness thresholds for your dataset.
3. Connect `CarlaSafetyController` to an actual CARLA actor.
4. Map the alert logic to your vehicle and simulator setup.
