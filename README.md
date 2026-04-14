# Drowsiness Detector + CARLA Safety Dashboard

Driver drowsiness monitoring with an optional CARLA-integrated safety controller and a tabbed PyQt5 dashboard (webcam + simulator view, live analytics, routing, and “park nearby” safety behavior).

This repository is intentionally built as a research/prototype-friendly scaffold:

- The **drowsiness state machine** is simple and easy to replace with a production model.
- The **dashboard** provides a real-time UI for monitoring and CARLA integration.
- The **CARLA controller** includes practical building blocks (sync mode, sensors, routing, shoulder parking helpers).

## Key Features

### Drowsiness detection

- **Drowsiness states**: `ALERT` → `DROWSY` → `SLEEPY`.
- **Two webcam backends**:
  - **Haar**: OpenCV Haar cascades (face + eyes) + a lightweight EAR-like heuristic.
  - **MediaPipe**: face mesh landmarks with smoothed EAR, plus yawn + head-nod heuristics.
- **Configurable thresholds** via `DrowsinessConfig` in `src/drowsiness_detector/config.py`.

### Alerts

- Cross-platform audible alert with layered fallbacks:
  - Windows `winsound`,
  - Linux `aplay` / `paplay` / `canberra-gtk-play`,
  - Qt `QApplication.beep()`,
  - terminal bell.

### CARLA safety controller (optional)

- Connects to CARLA, can **attach to an existing ego vehicle** by `role_name` or **spawn one**.
- Forces CARLA into **synchronous mode** (configurable fixed delta seconds).
- Spawns a **vehicle-mounted camera** with support for:
  - **RGB** camera (`sensor.camera.rgb`)
  - **Cosmos visualization** (`sensor.camera.cosmos_visualization`) with automatic fallback to RGB if the sensor isn’t available.
- Route planning + route following when CARLA agent modules are available.
- Safety actions:
  - applies **gentle slowdown** when `DROWSY`/`SLEEPY`,
  - requests a **safe stop** (“pull over”) when the detector hits its pull-over threshold,
  - can attempt **“park nearby”** shoulder-lane parking via route planning (with fallback to safe stop).

### Tabbed PyQt5 dashboard

- A single window with multiple tabs:
  - **Dashboard**: CARLA view + webcam view + minimap + live status chips.
  - **Monitoring**: annotated webcam view and EAR plot.
  - **Analytics**: speed/steer/throttle/brake time series.
  - **Parking**: dedicated parking camera feed + parking state timeline.
  - **Map**: click-to-set destination + plan/start/stop route controls.
- **Sleepy decision overlay**: after sustained `SLEEPY`, prompts “Continue to Destination” vs “Park Nearby”; auto-parks after timeout.
- Built-in **traffic generator toggle** (starts/stops CARLA’s `PythonAPI/examples/generate_traffic.py` when available).
- Keyboard controls for driving, camera switching, weather cycling, HUD/help toggles.

## Project Layout

- `src/drowsiness_detector/app.py`: CLI entrypoint (`--demo`, `--camera`, `--carla`)
- `src/drowsiness_detector/qt_dashboard.py`: PyQt5 dashboard runtime
- `src/drowsiness_detector/detector.py`: drowsiness state machine scaffold
- `src/drowsiness_detector/mediapipe_system.py`: MediaPipe-based drowsiness pipeline
- `src/drowsiness_detector/carla_controller.py`: CARLA connection, sensors, routing, safety maneuvers
- `src/drowsiness_detector/alerts.py`: audio/beep alert controller
- `tests/`: unit tests (state machine + controller behavior)

## Requirements

- Python **3.10+**
- Runtime packages are defined in `pyproject.toml` / `requirements.txt`.

Notes about CARLA:

- Many CARLA installs provide the Python API from the simulator distribution. If `import carla` fails, install/enable the CARLA Python API that matches your CARLA version.
- Some features (routing/agents/traffic generation) expect CARLA’s `PythonAPI/` folder to be present.

## Install

Create and activate a virtual environment, then install:

```bash
python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install -U pip
python3 -m pip install -e .
```

Alternatively:

```bash
python3 -m pip install -r requirements.txt
```

## Quickstart

This project exposes its CLI via `drowsiness_detector.app` (and `python -m drowsiness_detector` delegates to it).

### 1) Run the built-in demo stream

```bash
python3 -m drowsiness_detector --demo
```

### 2) Run the webcam dashboard (no CARLA)

```bash
python3 -m drowsiness_detector --camera --camera-index 0
```

Switch to the MediaPipe backend:

```bash
python3 -m drowsiness_detector --camera --drowsiness-backend mediapipe
```

### 3) Run webcam + CARLA dashboard

Start CARLA first, then:

```bash
python3 -m drowsiness_detector \
  --camera \
  --carla \
  --carla-host 127.0.0.1 \
  --carla-port 2000 \
  --carla-role-name hero
```

If there is no ego vehicle yet, spawn one automatically:

```bash
python3 -m drowsiness_detector \
  --camera \
  --carla \
  --spawn-vehicle-if-missing \
  --carla-role-name hero
```

Enable Cosmos visualization when available:

```bash
python3 -m drowsiness_detector \
  --camera \
  --carla \
  --spawn-vehicle-if-missing \
  --carla-role-name hero \
  --carla-camera-mode cosmos
```

## CLI Reference

Main modes:

- `--demo`: run a synthetic demo stream through the detector
- `--camera`: start the PyQt5 webcam dashboard
- `--carla`: connect the dashboard to CARLA

Common flags:

- `--camera-index 0`: webcam device index
- `--drowsiness-backend {haar,mediapipe}`: webcam backend
- `--carla-host`, `--carla-port`: CARLA connection
- `--carla-role-name hero`: attach to a vehicle with this role name
- `--spawn-vehicle-if-missing`: spawn an ego vehicle if none exist
- `--carla-map Town05`: load a specific CARLA map
- `--fixed-delta-seconds 0.05`: synchronous time step
- `--carla-camera-mode {rgb,cosmos}`: camera sensor mode
- `--carla-view-width/--carla-view-height`: CARLA preview resolution
- `--carla-view-index`: initial camera transform index
- `--log-level DEBUG`: logging verbosity (or set `LOG_LEVEL`)

## Dashboard Controls (CARLA mode)

The dashboard supports keyboard input when a CARLA vehicle is attached:

- Drive: `W/A/S/D` or arrow keys
- Brake: `S` / Down arrow
- Hand brake: `SPACE`
- Toggle route autopilot: `P` (requires a destination in the Map tab)
- Camera view: `TAB` next, `SHIFT+TAB` previous, `1`–`9` jump to view slot
- Secondary view shortcut: `Y` (high chase)
- Reverse: `Q`
- Manual transmission: `M`, then `,` / `.` for gear down/up
- Weather cycle: `C` (forward), `SHIFT+C` (backward)
- Lights cycle: `L` (position → low beam → fog → off)
- Toggle HUD: `F1`
- Toggle help: `H` or `/`
- Fullscreen: `F11`
- Quit: `ESC`

## Safety Behavior (CARLA mode)

When a CARLA vehicle is attached:

- `DROWSY`/`SLEEPY` applies a **slowdown** request via the controller.
- Sustained `SLEEPY` triggers a **decision overlay**:
  - “Continue to Destination” keeps/starts route following if a destination is set.
  - “Park Nearby” attempts to find a shoulder/parking lane segment and park via routing.
  - If the decision times out, it defaults to “Park Nearby”.
- If parking/routing fails, the system falls back to requesting a **safe stop**.

## Running Tests

```bash
python3 -m unittest discover -s tests
```

## Troubleshooting

### `PyQt5` / Qt plugin errors (Linux)

If Qt can’t find its platform plugins, ensure your environment can locate them. The dashboard attempts to set `QT_QPA_PLATFORM_PLUGIN_PATH` automatically, but system installs can vary.

### No webcam frames

- Try a different `--camera-index`.
- On Linux, check device permissions for `/dev/video*`.

### CARLA connects but you see no video

- Ensure the vehicle is attached/spawned and sensors are allowed by your CARLA build.
- If Cosmos sensor isn’t available, the app automatically falls back to RGB.

### Route/parking features unavailable

Some routing features require CARLA agent modules (e.g., `BasicAgent`) and a CARLA `PythonAPI/` checkout.

## VS Code Tasks

This workspace includes convenient tasks:

- “Run drowsiness demo”
- “Run CARLA webcam monitor”
- “Run detector tests”
