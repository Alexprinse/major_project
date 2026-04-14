from __future__ import annotations

import argparse
import logging
import os
import time

try:
    import carla  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    carla = None

from .alerts import AlertController
from .carla_controller import CarlaConnectionConfig, CarlaSafetyController
from .detector import DrowsinessDetector, DrowsinessState
from .qt_dashboard import run_camera_qt


def run_demo() -> None:
    detector = DrowsinessDetector()
    alerts = AlertController()
    carla_controller = CarlaSafetyController()

    sample_stream = [0.34, 0.28, 0.19, 0.17, 0.16, 0.15, 0.14, 0.13]
    for eye_aspect_ratio in sample_stream:
        state = detector.update(eye_aspect_ratio=eye_aspect_ratio, timestamp=time.time())
        print(f"sample={eye_aspect_ratio:.2f} state={state.value}")
        if state in {DrowsinessState.DROWSY, DrowsinessState.SLEEPY}:
            alerts.beep()
        if detector.should_pull_over and carla_controller.vehicle is not None:
            carla_controller.request_pull_over()
            print("Pull-over request sent to CARLA vehicle")
            break
        time.sleep(0.5)


def run_camera(
    camera_index: int = 0,
    carla_controller: CarlaSafetyController | None = None,
    carla_view_width: int = 960,
    carla_view_height: int = 540,
    carla_camera_mode: str = "rgb",
    carla_view_index: int = 1,
    drowsiness_backend: str = "haar",
) -> None:
    controller = carla_controller or CarlaSafetyController()
    print(f"Camera opened at index {camera_index}. Press ESC to quit.")
    print(f"Drowsiness backend: {drowsiness_backend}")
    run_camera_qt(
        camera_index=camera_index,
        carla_controller=controller,
        carla_view_width=carla_view_width,
        carla_view_height=carla_view_height,
        carla_camera_mode=carla_camera_mode,
        carla_view_index=carla_view_index,
        use_mediapipe=drowsiness_backend == "mediapipe",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drowsiness detection and CARLA safety scaffold")
    parser.add_argument("--demo", action="store_true", help="Run the built-in demo stream")
    parser.add_argument("--camera", action="store_true", help="Run the live webcam preview")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index to open")
    parser.add_argument("--carla", action="store_true", help="Connect to CARLA and attach to an ego vehicle")
    parser.add_argument("--carla-host", default="127.0.0.1", help="CARLA host")
    parser.add_argument("--carla-port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--carla-role-name", default="hero", help="Role name of the ego vehicle to attach to")
    parser.add_argument("--spawn-vehicle-if-missing", action="store_true", help="Spawn an ego vehicle if none exists")
    parser.add_argument("--carla-map", default=None, help="CARLA map name (e.g., Town05, Town03, etc.)")
    parser.add_argument("--carla-view-width", type=int, default=960, help="CARLA preview window width")
    parser.add_argument("--carla-view-height", type=int, default=540, help="CARLA preview window height")
    parser.add_argument("--sync", action="store_true", default=True, help="Run CARLA in synchronous mode (always on)")
    parser.add_argument(
        "--fixed-delta-seconds",
        type=float,
        default=0.05,
        help="Fixed simulation time-step used with --sync (default: 0.05)",
    )
    parser.add_argument(
        "--carla-camera-mode",
        choices=["rgb", "cosmos"],
        default="rgb",
        help="CARLA camera sensor mode: RGB or Cosmos visualization",
    )
    parser.add_argument(
        "--carla-view-index",
        type=int,
        default=3,
        help="CARLA camera transform index to use as the default view (0-8)",
    )
    parser.add_argument(
        "--drowsiness-backend",
        choices=["haar", "mediapipe"],
        default="haar",
        help="Drowsiness backend for webcam analysis",
    )

    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Python logging level (DEBUG, INFO, WARNING, ERROR). Can also be set via LOG_LEVEL env var.",
    )
    return parser.parse_args()


def _configure_logging(level: str) -> None:
    normalized = (level or "INFO").strip().upper()
    numeric_level = getattr(logging, normalized, logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    args = parse_args()
    _configure_logging(args.log_level)
    if args.demo:
        run_demo()
        return

    carla_controller: CarlaSafetyController | None = None
    if args.carla:
        carla_controller = CarlaSafetyController(
            connection=CarlaConnectionConfig(
                host=args.carla_host,
                port=args.carla_port,
                role_name=args.carla_role_name,
                map_name=args.carla_map,
            )
        )
        carla_controller.connect()
        carla_controller.enable_synchronous_mode(fixed_delta_seconds=args.fixed_delta_seconds)
        carla_controller.attach_or_spawn_vehicle(spawn_if_missing=args.spawn_vehicle_if_missing)
        print(
            f"Connected to CARLA at {args.carla_host}:{args.carla_port} and attached to vehicle role '{args.carla_role_name}'."
        )
        print(f"CARLA synchronous mode ON | fixed_delta_seconds={args.fixed_delta_seconds:.3f}")

    if args.camera:
        run_camera(
            camera_index=args.camera_index,
            carla_controller=carla_controller,
            carla_view_width=args.carla_view_width,
            carla_view_height=args.carla_view_height,
            carla_camera_mode=args.carla_camera_mode,
            carla_view_index=args.carla_view_index,
            drowsiness_backend=args.drowsiness_backend,
        )
        return
    print("No runtime mode selected. Use --demo, --camera, or --camera --carla to exercise the scaffold.")


if __name__ == "__main__":
    main()
