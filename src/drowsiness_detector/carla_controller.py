from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import os
import sys
from typing import Any

import numpy as np

try:
    import carla  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    carla = None

_AGENTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "PythonAPI", "carla")
)
if os.path.isdir(_AGENTS_PATH) and _AGENTS_PATH not in sys.path:
    sys.path.append(_AGENTS_PATH)

try:
    from agents.navigation.basic_agent import BasicAgent  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BasicAgent = None

try:
    from agents.navigation.local_planner import RoadOption  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    RoadOption = None

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    GlobalRoutePlanner = None


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class CarlaConnectionConfig:
    host: str = "127.0.0.1"
    port: int = 2000
    timeout_seconds: float = 5.0
    role_name: str = "hero"
    map_name: str | None = None


@dataclass
class CarlaSafetyController:
    """Minimal CARLA integration scaffold.

    Replace the placeholder methods with actor control and waypoint logic for
    your map, lane markings, and safe shoulder / parking area strategy.
    """

    connection: CarlaConnectionConfig = field(default_factory=CarlaConnectionConfig)
    client: Any | None = None
    world: Any | None = None
    vehicle: Any | None = None
    camera_sensor: Any | None = None
    rgb_reference_sensor: Any | None = None
    parking_camera_sensor: Any | None = None
    _latest_rgb_frame: np.ndarray | None = None
    _latest_rgb_reference_frame: np.ndarray | None = None
    _latest_parking_frame: np.ndarray | None = None
    _camera_width: int = 960
    _camera_height: int = 540
    _camera_fov: float = 90.0
    _camera_transform_index: int = 0
    _camera_transform_index_default: int = 0
    _secondary_camera_index: int = 3
    _parking_camera_transform_index: int = 7
    _camera_sensor_mode: str = "rgb"
    _camera_sensor_actual_mode: str = "rgb"
    _parking_agent: Any | None = None
    _parking_active: bool = False
    _parking_completed: bool = False
    _parking_mode: str = "none"
    _parking_phase: str = "idle"
    _parking_ticks_remaining: int = 0
    _parking_last_error: str = ""
    _parking_top_target_px: int = 80
    _parking_bottom_target_px: int = 140
    _parking_search_top_px: int = 6
    _parking_search_bottom_px: int = 100
    _parking_top_tolerance_px: int = 6
    _parking_bottom_tolerance_px: int = 8
    _parking_speed_multiplier: float = 2.0
    _parking_positioning_ticks: int = 0
    _parking_positioning_timeout_ticks: int = 320
    _parking_lane_change_start_key: tuple[int, int, int] | None = None
    _parking_lane_changed: bool = False
    _parking_lane_change_steer_max: float = 0.15
    _parking_min_positioning_ticks_before_rightmost_check: int = 20
    _parking_junction_cooldown_ticks: int = 0
    _parking_junction_cooldown_default_ticks: int = 35
    _parking_junction_agent_speed_kmh: float = 8.0
    _vehicle_spawned_by_controller: bool = False
    _sync_enabled: bool = False
    _fixed_delta_seconds: float = 0.05
    _original_world_settings: Any | None = None
    _traffic_manager: Any | None = None
    _map_waypoint_cache_key: tuple[str, float, bool, int | None] | None = None
    _map_waypoint_cache: list[dict] = field(default_factory=list)
    _route_agent: Any | None = None
    _route_destination: tuple[float, float, float] | None = None
    _route_polyline: list[tuple[float, float, float]] = field(default_factory=list)
    _route_planned: bool = False
    _route_active: bool = False
    _route_status: str = "idle"
    _route_arrival_distance_m: float = 3.0
    _route_reroute_distance_threshold_m: float = 12.0
    _route_reroute_required_ticks: int = 12
    _route_offtrack_counter: int = 0
    _route_reroute_cooldown_ticks: int = 0
    _route_reroute_cooldown_default_ticks: int = 30
    _route_parking_points: list[tuple[float, float, float]] = field(default_factory=list)
    _route_parking_active: bool = False
    _route_parking_index: int = 0
    _route_parking_arrival_distance_m: float = 1.2
    _route_parking_target_speed_kmh: float = 8.0
    _route_debug_step_counter: int = 0
    _route_parking_debug_step_counter: int = 0

    def available(self) -> bool:
        return carla is not None

    def connect(self) -> None:
        if carla is None:
            raise RuntimeError("CARLA Python API is not available. Install the carla package from your simulator build.")

        self.client = carla.Client(self.connection.host, self.connection.port)
        self.client.set_timeout(self.connection.timeout_seconds)
        
        if self.connection.map_name is not None:
            self.world = self.client.load_world(self.connection.map_name)
        else:
            self.world = self.client.get_world()

    def enable_synchronous_mode(self, fixed_delta_seconds: float = 0.05) -> None:
        if carla is None:
            raise RuntimeError("CARLA Python API is not available. Install the carla package from your simulator build.")
        if self.world is None:
            self.connect()
        if self.world is None:
            raise RuntimeError("CARLA world is not connected")

        settings = self.world.get_settings()
        self._original_world_settings = settings
        updated = self.world.get_settings()
        updated.synchronous_mode = True
        updated.fixed_delta_seconds = float(max(0.001, fixed_delta_seconds))
        self.world.apply_settings(updated)
        self._fixed_delta_seconds = updated.fixed_delta_seconds
        self._sync_enabled = True

        if self.client is not None:
            try:
                self._traffic_manager = self.client.get_trafficmanager()
                self._traffic_manager.set_synchronous_mode(True)
            except Exception:
                self._traffic_manager = None

    def disable_synchronous_mode(self) -> None:
        if self.world is None:
            return

        if self._traffic_manager is not None:
            try:
                self._traffic_manager.set_synchronous_mode(False)
            except Exception:
                pass
            self._traffic_manager = None

        if self._original_world_settings is not None:
            try:
                self.world.apply_settings(self._original_world_settings)
            except Exception:
                pass
        self._original_world_settings = None
        self._sync_enabled = False

    def tick(self) -> None:
        if self.world is None:
            return
        try:
            if self._sync_enabled:
                self.world.tick()
            else:
                self.world.wait_for_tick()
        except Exception:
            return

    def attach_vehicle(self, vehicle: Any | None = None) -> Any:
        if self.world is None:
            self.connect()

        if vehicle is not None:
            self.vehicle = vehicle
            self._vehicle_spawned_by_controller = False
            return vehicle

        if self.world is None:
            raise RuntimeError("CARLA world is not connected")

        actors = self.world.get_actors().filter("vehicle.*")
        if not actors:
            raise RuntimeError("No CARLA vehicles are available to attach")

        for actor in actors:
            role_name = actor.attributes.get("role_name") if hasattr(actor, "attributes") else None
            if role_name == self.connection.role_name:
                self.vehicle = actor
                self._vehicle_spawned_by_controller = False
                return actor

        self.vehicle = actors[0]
        self._vehicle_spawned_by_controller = False
        return self.vehicle

    def attach_or_spawn_vehicle(self, spawn_if_missing: bool = False, blueprint_filter: str = "vehicle.tesla.model3") -> Any:
        try:
            return self.attach_vehicle()
        except RuntimeError:
            if not spawn_if_missing:
                raise
            return self.spawn_vehicle(blueprint_filter=blueprint_filter)

    def spawn_vehicle(self, blueprint_filter: str = "vehicle.tesla.model3") -> Any:
        if carla is None:
            raise RuntimeError("CARLA Python API is not available. Install the carla package from your simulator build.")
        if self.world is None:
            self.connect()
        if self.world is None:
            raise RuntimeError("CARLA world is not connected")

        blueprint_library = self.world.get_blueprint_library()
        blueprints = blueprint_library.filter(blueprint_filter)
        if not blueprints:
            blueprints = blueprint_library.filter("vehicle.*")
        if not blueprints:
            raise RuntimeError("No vehicle blueprints are available in CARLA")

        blueprint = blueprints[0]
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", self.connection.role_name)

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available in the current CARLA map")

        spawned_actor = None
        for spawn_point in spawn_points:
            spawned_actor = self.world.try_spawn_actor(blueprint, spawn_point)
            if spawned_actor is not None:
                break

        if spawned_actor is None:
            raise RuntimeError("Unable to spawn vehicle: all spawn points appear occupied")

        self.vehicle = spawned_actor
        self._vehicle_spawned_by_controller = True
        return spawned_actor

    def start_rgb_preview(
        self,
        width: int = 960,
        height: int = 540,
        fov: float = 90.0,
        transform_index: int = 0,
        sensor_mode: str = "rgb",
    ) -> None:
        if carla is None:
            raise RuntimeError("CARLA Python API is not available. Install the carla package from your simulator build.")
        if self.world is None:
            self.connect()
        if self.vehicle is None:
            raise RuntimeError("No CARLA vehicle is attached to the safety controller")
        if self.world is None:
            raise RuntimeError("CARLA world is not connected")

        self._camera_width = width
        self._camera_height = height
        self._camera_fov = fov
        self._camera_transform_index = transform_index
        self._camera_transform_index_default = transform_index
        self._camera_sensor_mode = sensor_mode
        self._spawn_camera_sensor()

    def cycle_camera_transform(self, reverse: bool = False) -> int:
        if carla is None:
            raise RuntimeError("CARLA Python API is not available. Install the carla package from your simulator build.")

        transforms_count = len(self._camera_transforms())
        if transforms_count == 0:
            return 0

        if reverse:
            self._camera_transform_index = (self._camera_transform_index - 1) % transforms_count
        else:
            self._camera_transform_index = (self._camera_transform_index + 1) % transforms_count

        if self.camera_sensor is not None:
            self._spawn_camera_sensor()

        return self._camera_transform_index

    def set_camera_transform_index(self, index: int) -> int:
        transforms_count = len(self._camera_transforms())
        if transforms_count == 0:
            return 0

        self._camera_transform_index = index % transforms_count
        if self.camera_sensor is not None:
            self._spawn_camera_sensor()
        return self._camera_transform_index

    def cycle_to_secondary_view(self) -> int:
        """Switch to the secondary High chase view."""
        self._camera_transform_index = self._secondary_camera_index
        if self.camera_sensor is not None:
            self._spawn_camera_sensor()
        return self._camera_transform_index

    def reset_to_primary_view(self) -> int:
        """Reset camera to the default primary view."""
        self._camera_transform_index = self._camera_transform_index_default
        if self.camera_sensor is not None:
            self._spawn_camera_sensor()
        return self._camera_transform_index

    def get_camera_transform_index(self) -> int:
        return self._camera_transform_index

    def get_default_camera_index(self) -> int:
        return self._camera_transform_index_default

    def get_camera_view_name(self) -> str:
        labels = self.get_camera_view_labels()
        if not labels:
            return "Camera"
        return labels[self._camera_transform_index % len(labels)]

    def get_camera_mode(self) -> str:
        return self._camera_sensor_actual_mode

    def set_camera_mode(self, mode: str) -> str:
        normalized = mode.strip().lower()
        if normalized not in {"rgb", "cosmos"}:
            normalized = "rgb"
        self._camera_sensor_mode = normalized
        if self.camera_sensor is not None:
            self._spawn_camera_sensor()
        return self._camera_sensor_actual_mode

    def cycle_camera_mode(self) -> str:
        next_mode = "cosmos" if self._camera_sensor_mode == "rgb" else "rgb"
        return self.set_camera_mode(next_mode)

    def get_camera_view_labels(self) -> list[str]:
        return [
            "Rear spring arm",
            "Hood rigid",
            "Front spring arm",
            "High chase",
            "Windshield interior",
            "Front bumper",
            "Roof top-down",
            "Roof front chase",
            "Side rigid",
        ]

    def stop_rgb_preview(self) -> None:
        if self.camera_sensor is not None:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()
            self.camera_sensor = None
        if self.rgb_reference_sensor is not None:
            self.rgb_reference_sensor.stop()
            self.rgb_reference_sensor.destroy()
            self.rgb_reference_sensor = None
        if self.parking_camera_sensor is not None:
            self.parking_camera_sensor.stop()
            self.parking_camera_sensor.destroy()
            self.parking_camera_sensor = None
        self._latest_rgb_frame = None
        self._latest_rgb_reference_frame = None
        self._latest_parking_frame = None

    def destroy(self) -> None:
        self.stop_rgb_preview()
        self.disable_synchronous_mode()
        if self.vehicle is not None:
            try:
                if hasattr(self.vehicle, "set_autopilot"):
                    self.vehicle.set_autopilot(False)
            except Exception:
                pass
            try:
                self.vehicle.destroy()
            finally:
                self.vehicle = None
                self._vehicle_spawned_by_controller = False

    def get_latest_rgb_frame(self) -> np.ndarray | None:
        if self._latest_rgb_frame is None:
            return None
        return self._latest_rgb_frame.copy()

    def get_latest_rgb_reference_frame(self) -> np.ndarray | None:
        if self._latest_rgb_reference_frame is None:
            return None
        return self._latest_rgb_reference_frame.copy()

    def get_latest_parking_frame(self) -> np.ndarray | None:
        if self._latest_parking_frame is None:
            return None
        return self._latest_parking_frame.copy()

    def _on_camera_frame(self, image: Any) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        # CARLA camera raw_data arrives as BGRA; keep BGR for OpenCV pipeline.
        self._latest_rgb_frame = array[:, :, :3]

    def _on_rgb_reference_frame(self, image: Any) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self._latest_rgb_reference_frame = array[:, :, :3]

    def _on_parking_camera_frame(self, image: Any) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self._latest_parking_frame = array[:, :, :3]

    def _spawn_camera_sensor(self) -> None:
        if carla is None:
            raise RuntimeError("CARLA Python API is not available. Install the carla package from your simulator build.")
        if self.world is None or self.vehicle is None:
            raise RuntimeError("CARLA world and vehicle must be available before starting RGB preview")

        self.stop_rgb_preview()

        requested_blueprint_id = "sensor.camera.rgb" if self._camera_sensor_mode == "rgb" else "sensor.camera.cosmos_visualization"
        fallback_blueprint_id = "sensor.camera.rgb"

        blueprint_library = self.world.get_blueprint_library()
        try:
            blueprint = blueprint_library.find(requested_blueprint_id)
            self._camera_sensor_actual_mode = self._camera_sensor_mode
        except Exception:
            blueprint = blueprint_library.find(fallback_blueprint_id)
            self._camera_sensor_actual_mode = "rgb"

        blueprint.set_attribute("image_size_x", str(self._camera_width))
        blueprint.set_attribute("image_size_y", str(self._camera_height))
        blueprint.set_attribute("fov", str(self._camera_fov))

        transforms = self._camera_transforms()
        transform, attachment_type = transforms[self._camera_transform_index % len(transforms)]
        self.camera_sensor = self.world.spawn_actor(
            blueprint,
            transform,
            attach_to=self.vehicle,
            attachment_type=attachment_type,
        )
        self.camera_sensor.listen(self._on_camera_frame)

        # When active mode is Cosmos, also spawn a true RGB reference sensor.
        if self._camera_sensor_actual_mode == "cosmos":
            rgb_blueprint = blueprint_library.find("sensor.camera.rgb")
            rgb_blueprint.set_attribute("image_size_x", str(self._camera_width))
            rgb_blueprint.set_attribute("image_size_y", str(self._camera_height))
            rgb_blueprint.set_attribute("fov", str(self._camera_fov))
            self.rgb_reference_sensor = self.world.spawn_actor(
                rgb_blueprint,
                transform,
                attach_to=self.vehicle,
                attachment_type=attachment_type,
            )
            self.rgb_reference_sensor.listen(self._on_rgb_reference_frame)

        # Dedicated parking stream: always front view in cosmos (fallback to rgb).
        parking_transform, parking_attachment = transforms[self._parking_camera_transform_index % len(transforms)]
        try:
            parking_blueprint = blueprint_library.find("sensor.camera.cosmos_visualization")
        except Exception:
            parking_blueprint = blueprint_library.find("sensor.camera.rgb")
        parking_blueprint.set_attribute("image_size_x", str(self._camera_width))
        parking_blueprint.set_attribute("image_size_y", str(self._camera_height))
        parking_blueprint.set_attribute("fov", str(self._camera_fov))
        self.parking_camera_sensor = self.world.spawn_actor(
            parking_blueprint,
            parking_transform,
            attach_to=self.vehicle,
            attachment_type=parking_attachment,
        )
        self.parking_camera_sensor.listen(self._on_parking_camera_frame)

    def _camera_transforms(self) -> list[Any]:
        if carla is None:
            return []
        bound_x = 0.5
        bound_y = 0.5
        bound_z = 0.5
        if self.vehicle is not None:
            try:
                extent = self.vehicle.bounding_box.extent
                bound_x = 0.5 + extent.x
                bound_y = 0.5 + extent.y
                bound_z = 0.5 + extent.z
            except Exception:
                pass
        return [
            (
                carla.Transform(
                    carla.Location(x=-2.0 * bound_x, y=+0.0 * bound_y, z=2.0 * bound_z),
                    carla.Rotation(pitch=8.0),
                ),
                carla.AttachmentType.SpringArmGhost,
            ),
            (
                carla.Transform(carla.Location(x=+0.8 * bound_x, y=+0.0 * bound_y, z=1.3 * bound_z)),
                carla.AttachmentType.Rigid,
            ),
            (
                carla.Transform(
                    carla.Location(x=+1.9 * bound_x, y=+1.0 * bound_y, z=1.2 * bound_z),
                ),
                carla.AttachmentType.SpringArmGhost,
            ),
            (
                carla.Transform(
                    carla.Location(x=-2.8 * bound_x, y=+0.0 * bound_y, z=4.6 * bound_z),
                    carla.Rotation(pitch=6.0),
                ),
                carla.AttachmentType.SpringArmGhost,
            ),
            (
                carla.Transform(
                    carla.Location(x=+0.2 * bound_x, y=+0.0 * bound_y, z=1.1 * bound_z),
                    carla.Rotation(pitch=0.0),
                ),
                carla.AttachmentType.Rigid,
            ),
            (
                carla.Transform(
                    carla.Location(x=+2.1 * bound_x, y=+0.0 * bound_y, z=0.9 * bound_z),
                    carla.Rotation(pitch=-2.0),
                ),
                carla.AttachmentType.Rigid,
            ),
            (
                carla.Transform(
                    carla.Location(x=+0.0 * bound_x, y=+0.0 * bound_y, z=4.2 * bound_z),
                    carla.Rotation(pitch=-82.0, yaw=180.0),
                ),
                carla.AttachmentType.SpringArmGhost,
            ),
            (
                carla.Transform(
                    carla.Location(x=-0.0 * bound_x, y=+0.0 * bound_y, z=2.0 * bound_z),
                    carla.Rotation(pitch=-20.0),
                ),
                carla.AttachmentType.Rigid,
            ),
            (
                carla.Transform(carla.Location(x=-1.0, y=-1.0 * bound_y, z=0.4 * bound_z)),
                carla.AttachmentType.Rigid,
            ),
        ]

    def request_pull_over(self) -> None:
        if self.vehicle is None:
            raise RuntimeError("No CARLA vehicle is attached to the safety controller")
        self.vehicle.apply_control(self._build_safe_stop_control())

    def inject_parking_waypoints(
        self,
        points: list[tuple[float, float, float]],
        *,
        activate: bool = True,
        reason: str = "parking-waypoints",
    ) -> tuple[bool, str]:
        """Inject waypoints into the existing parking waypoint controller.

        This reuses `_run_parking_waypoint_step()` by populating `_route_parking_points`
        and activating the parking follower immediately.
        """
        if carla is None or self.world is None or self.vehicle is None:
            return False, "CARLA world/vehicle unavailable"

        normalized = [tuple(float(v) for v in point) for point in points if point is not None]
        if len(normalized) < 1:
            return False, "No parking waypoints provided"

        try:
            self.set_autopilot_enabled(False)
        except Exception:
            pass

        self._route_destination = normalized[-1]
        self._route_polyline = list(normalized)
        self._route_planned = True
        self._route_active = True
        self._route_agent = None
        self._route_offtrack_counter = 0
        self._route_reroute_cooldown_ticks = 0
        self._route_debug_step_counter = 0

        self._route_parking_points = list(normalized)
        self._route_parking_active = bool(activate)
        self._route_parking_index = 0
        self._route_status = "parking-running" if activate else "planned"
        return True, f"Injected {len(normalized)} parking waypoints ({reason})"

    def start_nearest_shoulder_lane_parking(
        self,
        *,
        min_consecutive_waypoints: int = 4,
        stop_ahead_waypoints: int = 5,
        waypoint_spacing_m: float = 2.0,
        scan_steps: int = 40,
        avoid_junctions: bool = True,
    ) -> tuple[bool, str]:
        """Find the nearest valid shoulder-lane parking spot and park there.

        A valid spot is `min_consecutive_waypoints` consecutive *shoulder* waypoints.
        """
        if carla is None or self.world is None or self.vehicle is None:
            return False, "CARLA world/vehicle unavailable"

        desired_points = max(int(max(1, min_consecutive_waypoints)), 1 + int(max(1, stop_ahead_waypoints)))
        segment = self._find_nearest_consecutive_shoulder_waypoints(
            min_points=int(max(1, min_consecutive_waypoints)),
            desired_points=int(desired_points),
            step_distance=float(max(0.5, waypoint_spacing_m)),
            scan_steps=int(max(1, scan_steps)),
            avoid_junctions=bool(avoid_junctions),
        )
        if len(segment) < int(max(1, min_consecutive_waypoints)):
            return False, "No valid shoulder parking spot found"

        return self.inject_parking_waypoints(segment, activate=True, reason="shoulder-parking")

    def start_nearest_shoulder_lane_parking_via_route(
        self,
        *,
        min_consecutive_waypoints: int = 4,
        stop_ahead_waypoints: int = 5,
        waypoint_spacing_m: float = 2.0,
        scan_steps: int = 40,
        target_speed_kmh: float = 15.0,
        reason: str = "shoulder-parking-via-route",
        avoid_junctions: bool = True,
        require_ahead: bool = True,
        search_after_next_junction_if_needed: bool = True,
    ) -> tuple[bool, str]:
        """Park on a nearby shoulder lane using the normal route-following pipeline.

        Unlike `start_nearest_shoulder_lane_parking()`, this does NOT immediately
        switch into the custom parking waypoint follower.

        Instead it:
        1) Finds a valid shoulder segment near the vehicle.
        2) Uses a point `stop_ahead_waypoints` ahead on that segment as the route destination.
        3) Plans a route and starts `BasicAgent` driving, allowing the existing
           lane-change transition + parking handoff logic to run.

        This produces smoother steering when entering the shoulder/parking lane.
        """
        if carla is None or self.world is None or self.vehicle is None:
            return False, "CARLA world/vehicle unavailable"

        try:
            current_location = self.vehicle.get_location()
            LOGGER.info(
                "[shoulder-route] start reason=%s location=(%.2f, %.2f, %.2f)",
                str(reason),
                float(current_location.x),
                float(current_location.y),
                float(current_location.z),
            )
        except Exception:
            LOGGER.info("[shoulder-route] start reason=%s", str(reason))

        desired_points = max(int(max(1, min_consecutive_waypoints)), 1 + int(max(1, stop_ahead_waypoints)))

        # Prefer forward-only shoulder targets to avoid selecting a "nearby" point
        # that is behind/side of the vehicle and requires a loop to reach.
        segment = self._find_nearest_consecutive_shoulder_waypoints_forward(
            min_points=int(max(1, min_consecutive_waypoints)),
            desired_points=int(desired_points),
            step_distance=float(max(0.5, waypoint_spacing_m)),
            scan_steps=int(max(1, scan_steps)),
            avoid_junctions=bool(avoid_junctions),
            stop_before_junction=True,
            require_ahead=bool(require_ahead),
        )

        if not segment and bool(search_after_next_junction_if_needed):
            segment = self._find_nearest_consecutive_shoulder_waypoints_forward(
                min_points=int(max(1, min_consecutive_waypoints)),
                desired_points=int(desired_points),
                step_distance=float(max(0.5, waypoint_spacing_m)),
                scan_steps=int(max(1, scan_steps)),
                avoid_junctions=bool(avoid_junctions),
                start_after_junction=True,
                require_ahead=bool(require_ahead),
            )

        if len(segment) < int(max(1, min_consecutive_waypoints)):
            LOGGER.info(
                "[shoulder-route] no-segment reason=%s min_points=%d scan_steps=%d",
                str(reason),
                int(max(1, min_consecutive_waypoints)),
                int(max(1, scan_steps)),
            )
            return False, "No valid shoulder parking spot found"

        target_index = int(max(0, min(int(stop_ahead_waypoints), len(segment) - 1)))
        target = segment[target_index]

        if bool(require_ahead) and not self._is_point_ahead_of_vehicle(target, min_forward_dot=0.05):
            LOGGER.info(
                "[shoulder-route] rejected target not-ahead target=(%.2f, %.2f, %.2f)",
                float(target[0]),
                float(target[1]),
                float(target[2]),
            )
            return False, "Selected shoulder target was not ahead of vehicle"

        LOGGER.info(
            "[shoulder-route] selected segment_points=%d target_index=%d target=(%.2f, %.2f, %.2f)",
            len(segment),
            int(target_index),
            float(target[0]),
            float(target[1]),
            float(target[2]),
        )

        # Cancel any immediate parking follower state.
        self._route_parking_active = False
        self._route_parking_index = 0

        ok, msg = self.set_destination_location(*target)
        if not ok:
            return False, msg

        planned, plan_msg = self.plan_route_to_destination()
        if not planned:
            LOGGER.warning("[shoulder-route] plan-failed reason=%s error=%s", str(reason), str(plan_msg))
            return False, plan_msg

        started, follow_msg = self.start_route_following(target_speed_kmh=float(max(1.0, target_speed_kmh)))
        if not started:
            LOGGER.warning("[shoulder-route] follow-start-failed reason=%s error=%s", str(reason), str(follow_msg))
            return False, follow_msg

        LOGGER.info(
            "[shoulder-route] follow-started reason=%s target_speed_kmh=%.1f",
            str(reason),
            float(max(1.0, target_speed_kmh)),
        )
        return True, f"Shoulder parking route started (target_index={target_index}): {follow_msg}"

    def _find_nearest_consecutive_shoulder_waypoints(
        self,
        *,
        min_points: int = 4,
        desired_points: int | None = None,
        step_distance: float = 2.0,
        scan_steps: int = 40,
        avoid_junctions: bool = True,
    ) -> list[tuple[float, float, float]]:
        if carla is None or self.world is None or self.vehicle is None:
            return []

        world_map = self.world.get_map()
        current_wp = world_map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Parking,
        )
        if current_wp is None:
            return []

        # Prefer the closest candidates, checking forward before backward.
        candidates = [current_wp]
        for step in range(1, int(max(1, scan_steps)) + 1):
            dist = float(step) * float(step_distance)
            nexts = current_wp.next(dist)
            if nexts:
                candidates.append(nexts[0])
            prevs = current_wp.previous(dist)
            if prevs:
                candidates.append(prevs[0])

        desired = int(max(1, desired_points)) if desired_points is not None else int(max(1, min_points))

        for driving_wp in candidates:
            if bool(avoid_junctions) and self._waypoint_is_in_junction(driving_wp):
                continue
            shoulder_seed = self._first_right_lane_of_type(driving_wp, lane_type=carla.LaneType.Shoulder)
            if shoulder_seed is None:
                continue
            if bool(avoid_junctions) and self._waypoint_is_in_junction(shoulder_seed):
                continue
            segment = self._build_consecutive_lane_segment(
                shoulder_seed,
                lane_type=carla.LaneType.Shoulder,
                count=int(max(1, desired)),
                step_distance=float(step_distance),
                avoid_junctions=bool(avoid_junctions),
            )
            if len(segment) >= int(max(1, min_points)):
                return segment
        return []

    def _find_nearest_consecutive_shoulder_waypoints_forward(
        self,
        *,
        min_points: int = 4,
        desired_points: int | None = None,
        step_distance: float = 2.0,
        scan_steps: int = 40,
        avoid_junctions: bool = True,
        stop_before_junction: bool = False,
        start_after_junction: bool = False,
        require_ahead: bool = True,
    ) -> list[tuple[float, float, float]]:
        """Forward-only shoulder segment search.

        This is used for sleepy parking to avoid selecting a shoulder point that is
        physically close but behind/side of the vehicle (which can cause looping routes).

        - If `stop_before_junction` is True, scanning stops when entering a junction.
        - If `start_after_junction` is True, scanning skips everything until we've
          crossed a junction (enter -> exit) and then begins searching.
        """
        if carla is None or self.world is None or self.vehicle is None:
            return []

        world_map = self.world.get_map()
        current_wp = world_map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Parking,
        )
        if current_wp is None:
            return []

        desired = int(max(1, desired_points)) if desired_points is not None else int(max(1, min_points))
        driving_candidates: list[Any] = []

        cursor = current_wp
        saw_junction = False
        exited_junction = False

        for _ in range(int(max(1, scan_steps))):
            in_junction = self._waypoint_is_in_junction(cursor)
            if in_junction:
                saw_junction = True
                if bool(stop_before_junction):
                    break
            else:
                if saw_junction:
                    exited_junction = True

            if bool(start_after_junction) and not exited_junction:
                # Keep scanning forward until we have crossed a junction.
                pass
            else:
                driving_candidates.append(cursor)

            nexts = cursor.next(float(step_distance))
            if not nexts:
                break
            cursor = nexts[0]

        # If we're required to start after a junction but never crossed one, there is
        # no valid forward target under this mode.
        if bool(start_after_junction) and not exited_junction:
            return []

        for driving_wp in driving_candidates:
            if bool(avoid_junctions) and self._waypoint_is_in_junction(driving_wp):
                continue

            shoulder_seed = self._first_right_lane_of_type(driving_wp, lane_type=carla.LaneType.Shoulder)
            if shoulder_seed is None:
                continue
            if bool(avoid_junctions) and self._waypoint_is_in_junction(shoulder_seed):
                continue

            segment = self._build_consecutive_lane_segment(
                shoulder_seed,
                lane_type=carla.LaneType.Shoulder,
                count=int(max(1, desired)),
                step_distance=float(step_distance),
                avoid_junctions=bool(avoid_junctions),
            )
            if len(segment) < int(max(1, min_points)):
                continue
            if bool(require_ahead):
                target_idx = min(len(segment) - 1, max(0, int(desired) - 1))
                if not self._is_point_ahead_of_vehicle(segment[target_idx], min_forward_dot=0.05):
                    continue
            return segment
        return []

    def _is_point_ahead_of_vehicle(
        self,
        point_xyz: tuple[float, float, float],
        *,
        min_forward_dot: float = 0.0,
    ) -> bool:
        if self.vehicle is None:
            return True
        try:
            transform = self.vehicle.get_transform()
            yaw_rad = math.radians(float(transform.rotation.yaw))
            forward_x = math.cos(yaw_rad)
            forward_y = math.sin(yaw_rad)
            vec_x = float(point_xyz[0]) - float(transform.location.x)
            vec_y = float(point_xyz[1]) - float(transform.location.y)
            norm = math.sqrt(vec_x * vec_x + vec_y * vec_y)
            if norm < 1e-6:
                return True
            dot = (vec_x / norm) * forward_x + (vec_y / norm) * forward_y
            return float(dot) >= float(min_forward_dot)
        except Exception:
            return True

    def _waypoint_is_in_junction(self, waypoint: Any | None) -> bool:
        if waypoint is None:
            return False
        try:
            value = getattr(waypoint, "is_junction")
            if callable(value):
                return bool(value())
            return bool(value)
        except Exception:
            return False

    def _first_right_lane_of_type(self, waypoint, *, lane_type) -> Any | None:
        if carla is None or waypoint is None:
            return None
        try:
            if waypoint.lane_type == lane_type:
                return waypoint
        except Exception:
            return None

        cursor = waypoint
        for _ in range(6):
            try:
                cursor = cursor.get_right_lane()
            except Exception:
                cursor = None
            if cursor is None:
                return None
            try:
                if cursor.lane_type == lane_type:
                    return cursor
            except Exception:
                return None
        return None

    def _build_consecutive_lane_segment(
        self,
        seed_waypoint,
        *,
        lane_type,
        count: int,
        step_distance: float,
        avoid_junctions: bool = True,
    ) -> list[tuple[float, float, float]]:
        if seed_waypoint is None:
            return []
        if carla is None:
            return []

        segment: list[tuple[float, float, float]] = []
        cursor = seed_waypoint

        for _ in range(int(max(1, count))):
            try:
                if cursor.lane_type != lane_type:
                    return []
                if bool(avoid_junctions) and self._waypoint_is_in_junction(cursor):
                    return []
                loc = cursor.transform.location
                segment.append((float(loc.x), float(loc.y), float(loc.z)))

                nexts = cursor.next(float(step_distance))
                if not nexts:
                    break
                same_lane = [wp for wp in nexts if wp.lane_type == lane_type and wp.lane_id == cursor.lane_id]
                cursor = same_lane[0] if same_lane else nexts[0]
            except Exception:
                return []

        return segment

    def set_autopilot_enabled(self, enabled: bool) -> bool:
        if self.vehicle is None:
            return False
        try:
            tm_port = None
            if self._traffic_manager is not None and hasattr(self._traffic_manager, "get_port"):
                tm_port = self._traffic_manager.get_port()
            if tm_port is not None:
                self.vehicle.set_autopilot(enabled, tm_port)
            else:
                self.vehicle.set_autopilot(enabled)
            return True
        except Exception:
            return False

    def set_destination_location(self, x: float, y: float, z: float = 0.0) -> tuple[bool, str]:
        if carla is None or self.world is None:
            return False, "CARLA world unavailable"

        try:
            desired = carla.Location(x=float(x), y=float(y), z=float(z))
            waypoint = self.world.get_map().get_waypoint(
                desired,
                project_to_road=True,
                lane_type=carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Parking,
            )
            if waypoint is None:
                return False, "No road waypoint near selected destination"

            loc = waypoint.transform.location
            self._route_destination = (float(loc.x), float(loc.y), float(loc.z))
            self._route_polyline = []
            self._route_planned = False
            self._route_active = False
            self._route_agent = None
            self._route_status = "destination-selected"
            self._route_offtrack_counter = 0
            self._route_reroute_cooldown_ticks = 0
            self._route_parking_points = []
            self._route_parking_active = False
            self._route_parking_index = 0
            return True, f"Destination set at x={loc.x:.1f}, y={loc.y:.1f}, z={loc.z:.1f}"
        except Exception as exc:
            return False, f"Failed to set destination: {exc}"

    def clear_destination(self) -> None:
        self._route_destination = None
        self._route_polyline = []
        self._route_planned = False
        self._route_active = False
        self._route_agent = None
        self._route_status = "idle"
        self._route_offtrack_counter = 0
        self._route_reroute_cooldown_ticks = 0
        self._route_parking_points = []
        self._route_parking_active = False
        self._route_parking_index = 0

    @staticmethod
    def _distance_xyz(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        dz = float(a[2]) - float(b[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _plan_parking_waypoint_segment(
        self,
        start: tuple[float, float, float],
        goal_hint: tuple[float, float, float],
        resolution: float = 2.0,
    ) -> list[tuple[float, float, float]]:

        # AFTER (fixed) ────────────────────────────────────────────────────────────────
        candidates = self._collect_parking_shoulder_waypoints(
            goal_hint=goal_hint, search_radius_m=80.0
        )

        LOGGER.info(
            "[route] parking-segment: candidate_count=%d resolution=%.2f start=%s goal=%s",
            len(candidates),
            float(resolution),
            tuple(round(float(v), 2) for v in start),
            tuple(round(float(v), 2) for v in goal_hint),
        )

        if not candidates:
            LOGGER.warning("[route] parking-segment: no parking/shoulder candidates found")
            return []

        def angle(a, b, c):
            ab = np.array(b) - np.array(a)
            bc = np.array(c) - np.array(b)
            if np.linalg.norm(ab) < 1e-3 or np.linalg.norm(bc) < 1e-3:
                return 0.0
            cos_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
            return np.arccos(np.clip(cos_angle, -1.0, 1.0))

        start_idx = min(range(len(candidates)), key=lambda i: self._distance_xyz(candidates[i], start))
        goal_idx = min(range(len(candidates)), key=lambda i: self._distance_xyz(candidates[i], goal_hint))

        current_idx = start_idx
        visited = {start_idx}
        path = [candidates[start_idx]]

        for _ in range(80):
            if current_idx == goal_idx:
                break

            current = candidates[current_idx]

            neighbors = sorted(
                (
                    idx for idx in range(len(candidates))
                    if idx not in visited and self._distance_xyz(candidates[idx], current) > 0.2
                ),
                key=lambda idx: self._distance_xyz(candidates[idx], current)
            )[:15]

            if not neighbors:
                break

            best_idx = None
            best_score = float("inf")

            for idx in neighbors:
                next_pt = candidates[idx]

                dist_goal = self._distance_xyz(next_pt, candidates[goal_idx])
                dist_step = self._distance_xyz(current, next_pt)

                angle_penalty = 0.0
                if len(path) >= 2:
                    angle_penalty = angle(path[-2], path[-1], next_pt)

                score = dist_goal + 0.5 * dist_step + 3.0 * angle_penalty

                if score < best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                break

            visited.add(best_idx)
            current_idx = best_idx
            path.append(candidates[current_idx])

        # ensure final goal
        if self._distance_xyz(path[-1], candidates[goal_idx]) > 2.0:
            path.append(candidates[goal_idx])

        LOGGER.info(
            "[route] parking-segment: built_path_points=%d start_idx=%d goal_idx=%d",
            len(path),
            int(start_idx),
            int(goal_idx),
        )

        return path

    def plan_route_to_destination(self, sampling_resolution: float = 2.0) -> tuple[bool, str]:

        if carla is None or self.world is None or self.vehicle is None:
            return False, "CARLA world/vehicle unavailable"

        if self._route_destination is None:
            return False, "No destination selected"

        route_points: list[tuple[float, float, float]] = []

        try:
            start_location = self.vehicle.get_location()
            destination_location = carla.Location(*self._route_destination)
            LOGGER.info(
                "[route] plan-start: start=(%.2f, %.2f, %.2f) destination=(%.2f, %.2f, %.2f)",
                float(start_location.x),
                float(start_location.y),
                float(start_location.z),
                float(destination_location.x),
                float(destination_location.y),
                float(destination_location.z),
            )

            # -------------------------------------------------
            # 1. NORMAL ROUTE (Global Planner)
            # -------------------------------------------------
            if GlobalRoutePlanner is not None:
                planner = GlobalRoutePlanner(
                    self.world.get_map(),
                    max(0.5, float(sampling_resolution))
                )

                trace = planner.trace_route(start_location, destination_location)

                for waypoint, _ in trace:
                    loc = waypoint.transform.location
                    route_points.append((float(loc.x), float(loc.y), float(loc.z)))

            # fallback if planner fails
            if not route_points:
                route_points = [
                    (float(start_location.x), float(start_location.y), float(start_location.z)),
                    self._route_destination,
                ]

            # -------------------------------------------------
            # 2. CHECK IF PARKING IS NEEDED
            # -------------------------------------------------
            use_parking = self._is_destination_near_parking(self._route_destination)
            LOGGER.info("[route] parking-check: use_parking=%s", bool(use_parking))

            if use_parking:

                # -------------------------------------------------
                # 3. FIND ENTRY POINT (7 WAYPOINTS BEFORE DESTINATION)
                # -------------------------------------------------
                entry_wp = self._get_waypoint_n_steps_back(
                    self._route_destination,
                    steps=7,
                    step_distance=2.0
                )
                LOGGER.info("[route] parking-entry-base: found=%s", entry_wp is not None)

                if entry_wp is not None:

                    # -------------------------------------------------
                    # 4. SHIFT TO RIGHT (PARKING LANE)
                    # -------------------------------------------------
                    parking_entry_wp = self._get_right_lane_entry_point(entry_wp)
                    LOGGER.info("[route] parking-entry-right-lane: found=%s", parking_entry_wp is not None)

                    if parking_entry_wp is not None:

                        # To avoid a sharp lateral hop into the parking/shoulder lane,
                        # move the entry target a few waypoints forward along that lane.
                        # (Roughly 5 waypoints * 2 m ≈ 10 m by default.)
                        try:
                            entry_advance_steps = 3
                            cursor = parking_entry_wp
                            for _ in range(int(max(0, entry_advance_steps))):
                                next_wps = cursor.next(2.0)
                                if not next_wps:
                                    break
                                same_lane = [
                                    wp
                                    for wp in next_wps
                                    if wp.lane_id == cursor.lane_id and wp.lane_type == cursor.lane_type
                                ]
                                cursor = same_lane[0] if same_lane else next_wps[0]
                            parking_entry_wp = cursor
                        except Exception:
                            pass

                        entry_loc = parking_entry_wp.transform.location
                        entry_point = (
                            float(entry_loc.x),
                            float(entry_loc.y),
                            float(entry_loc.z)
                        )

                        # -------------------------------------------------
                        # 5. REMOVE LAST FEW POINTS (avoid sharp turn)
                        # -------------------------------------------------
                        if len(route_points) > 7:
                            route_points = route_points[:-7]

                        # -------------------------------------------------
                        # 6. SMOOTH TRANSITION (lane change)
                        # -------------------------------------------------
                        transition = self._generate_lane_change_transition(
                            route_points[-1],
                            entry_point,
                            steps=7
                        )

                        route_points.extend(transition)

                        # -------------------------------------------------
                        # 7. GENERATE FULL PARKING LANE PATH
                        # -------------------------------------------------

                        parking_path = self._plan_parking_waypoint_segment(
                            start=entry_point,
                            goal_hint=self._route_destination,
                            resolution=2.0
                        )

                        if parking_path:
                            route_points.extend(parking_path)
                            LOGGER.info(
                                "[route] parking-path: appended_points=%d entry_point=%s",
                                len(parking_path),
                                tuple(round(float(v), 2) for v in entry_point),
                            )
                        else:
                            # fallback (if algo fails)
                            route_points.append(entry_point)
                            route_points.append(self._route_destination)
                            LOGGER.warning("[route] parking-path: algorithm failed, using fallback entry+destination only")

                        # store for parking controller
                        self._route_parking_points = parking_path if parking_path else []
                        LOGGER.info(
                            "[route] parking-controller-points: count=%d",
                            len(self._route_parking_points),
                        )

                    else:
                        self._route_parking_points = []
                        LOGGER.warning("[route] parking mode requested but no right parking/shoulder lane found")

                else:
                    self._route_parking_points = []
                    LOGGER.warning("[route] parking mode requested but could not find driving entry waypoint")

            else:
                self._route_parking_points = []

            # -------------------------------------------------
            # 8. FINALIZE ROUTE
            # -------------------------------------------------
            self._route_polyline = route_points
            self._route_planned = True
            self._route_status = "planned"
            self._route_offtrack_counter = 0
            self._route_reroute_cooldown_ticks = 0
            self._route_debug_step_counter = 0

            LOGGER.info(
                "[route] plan-finished: total_route_points=%d parking_points=%d",
                len(self._route_polyline),
                len(self._route_parking_points),
            )

            return True, f"Route planned with {len(route_points)} points"

        except Exception as exc:
            self._route_planned = False
            self._route_status = "failed"
            self._route_parking_points = []
            return False, f"Route planning failed: {exc}"      
          
    def _generate_lane_change_transition(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        steps: int = 5
    ) -> list[tuple[float, float, float]]:
        """
        Generate smooth intermediate points for lane change.
        """
        transition = []

        for i in range(1, steps + 1):
            alpha = i / steps
            x = (1 - alpha) * start[0] + alpha * end[0]
            y = (1 - alpha) * start[1] + alpha * end[1]
            z = (1 - alpha) * start[2] + alpha * end[2]
            transition.append((x, y, z))

        return transition

    def _is_destination_near_parking(self, dest, threshold: float = 3.0) -> bool:
        """
        Check if destination lies on parking or shoulder lane (NOT driving lane).
        """

        if carla is None or self.world is None:
            return False

        waypoint = self.world.get_map().get_waypoint(
            carla.Location(*dest),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
            | carla.LaneType.Shoulder
            | carla.LaneType.Parking,
        )

        if waypoint is None:
            return False

        # 🚨 ONLY consider real parking/shoulder lanes
        return waypoint.lane_type in (
            carla.LaneType.Parking,
            carla.LaneType.Shoulder,
        )
    
    def _get_waypoint_n_steps_back(self, location, steps=7, step_distance=2.0):
        """
        Move backward along lane by N steps.
        """
        if carla is None or self.world is None:
            return None

        waypoint = self.world.get_map().get_waypoint(
            carla.Location(*location),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if waypoint is None:
            return None

        current = waypoint

        for _ in range(steps):
            prev_wps = current.previous(step_distance)
            if not prev_wps:
                break
            current = prev_wps[0]

        return current
    
    def _get_right_lane_entry_point(self, waypoint):
        """
        Move to the first RIGHT lane that is actually parking/shoulder.
        """

        if waypoint is None:
            return None

        wp = waypoint

        while True:
            right = wp.get_right_lane()
            if right is None:
                break

            # ✅ stop ONLY when real parking/shoulder found
            if right.lane_type in (carla.LaneType.Parking, carla.LaneType.Shoulder):
                return right

            wp = right

        return None

    def start_route_following(
        self,
        target_speed_kmh: float = 30.0,
        *,
        force_replan: bool = False,
    ) -> tuple[bool, str]:
        """
        Starts route following using BasicAgent.

        Fixes:
        1. Uses GLOBAL PLAN instead of just set_destination
        2. Ensures agent follows YOUR generated waypoints
        """

        if carla is None or self.vehicle is None:
            return False, "Vehicle unavailable"

        if self.world is None:
            return False, "CARLA world unavailable"

        if BasicAgent is None:
            self._route_status = "failed"
            return False, "BasicAgent unavailable"

        if self._route_destination is None:
            return False, "No destination selected"

        # If resuming after manual takeover, the previous plan may be stale.
        # When requested (or when far away from the cached polyline), replan from
        # the vehicle's *current* position so the agent can recover.
        if force_replan:
            self._route_planned = False
        elif self._route_planned and self._route_polyline:
            try:
                distance_to_route = self._distance_to_route_m(self.vehicle.get_location())
            except Exception:
                distance_to_route = None

            if (
                distance_to_route is not None
                and distance_to_route > float(self._route_reroute_distance_threshold_m)
            ):
                LOGGER.info(
                    "[route] start: cached route too far (dist=%.2f m) -> replanning",
                    float(distance_to_route),
                )
                self._route_planned = False

        # 🔥 Ensure route is planned
        if not self._route_planned or not self._route_polyline:
            planned, message = self.plan_route_to_destination()
            if not planned:
                return False, message

        try:
            self.set_autopilot_enabled(False)

            # 🔥 Initialize agent
            self._route_agent = BasicAgent(self.vehicle)

            # Set speed
            if hasattr(self._route_agent, "set_target_speed"):
                self._route_agent.set_target_speed(float(target_speed_kmh))

            # -------------------------------------------------
            # Use an explicit global plan when RoadOption is available.
            # -------------------------------------------------
            # AFTER (fixed) ────────────────────────────────────────────────────────────────
            # Only feed the DRIVING segment to BasicAgent.
            # Parking points are handled exclusively by _run_parking_waypoint_step().
            driving_polyline = (
                self._route_polyline[: -len(self._route_parking_points)]
                if self._route_parking_points
                else self._route_polyline
            )

            plan = []
            for point in driving_polyline:
                wp = self.world.get_map().get_waypoint(
                    carla.Location(*point),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,   # ← driving lane ONLY for BasicAgent
                )
                if wp is not None:
                    if RoadOption is not None:
                        plan.append((wp, RoadOption.LANEFOLLOW))
                    else:
                        plan.append(wp)

            if not plan:
                return False, "Generated plan is empty"

            if RoadOption is not None and hasattr(self._route_agent, "set_global_plan"):
                self._route_agent.set_global_plan(plan)
            elif hasattr(self._route_agent, "set_destination"):
                # Fall back: aim for the parking entry point, not the final destination
                entry = driving_polyline[-1] if driving_polyline else self._route_destination
                self._route_agent.set_destination(carla.Location(*entry))
            else:
                return False, "Route agent does not support global plan or destination setting"

            # -------------------------------------------------

            self._route_active = True
            self._route_status = "running"
            self._route_offtrack_counter = 0
            self._route_reroute_cooldown_ticks = 0
            self._route_parking_active = False
            self._route_parking_index = 0
            self._route_debug_step_counter = 0

            LOGGER.info(
                "[route] follow-start: plan_waypoints=%d parking_points=%d parking_active=%s target_speed_kmh=%.1f",
                len(plan),
                len(self._route_parking_points),
                self._route_parking_active,
                float(target_speed_kmh),
            )
            if self._route_parking_points:
                LOGGER.info("[route] note: parking points exist but parking controller is inactive until explicitly enabled")

            return True, f"Route following started with {len(plan)} waypoints"

        except Exception as exc:
            self._route_agent = None
            self._route_active = False
            self._route_status = "failed"
            return False, f"Could not start route following: {exc}"

    def stop_route_following(self, reason: str = "stopped") -> None:
        self._route_active = False
        self._route_agent = None
        self._route_status = reason
        self._route_offtrack_counter = 0
        self._route_reroute_cooldown_ticks = 0
        self._route_parking_active = False
        self._route_parking_index = 0
        self._route_parking_debug_step_counter = 0

    def _run_parking_waypoint_step(self) -> str:
        if carla is None or self.vehicle is None:
            self._route_status = "failed"
            self._route_active = False
            self._route_parking_active = False
            return self._route_status
        if not self._route_parking_points:
            LOGGER.warning("[route] parking-step called with zero parking points; marking arrived")
            self._route_status = "arrived"
            self._route_active = False
            self._route_parking_active = False
            return self._route_status

        self._route_parking_index = max(0, min(self._route_parking_index, len(self._route_parking_points) - 1))

        self._route_parking_debug_step_counter += 1
        current_location = self.vehicle.get_location()
        target = self._route_parking_points[self._route_parking_index]
        dist_to_target = self._distance_xyz(
            (float(current_location.x), float(current_location.y), float(current_location.z)),
            target,
        )

        if self._route_parking_debug_step_counter % 10 == 0:
            try:
                speed_kmh = self.get_speed_kmh()
            except Exception:
                speed_kmh = 0.0
            LOGGER.debug(
                "[route] parking-step tick=%d index=%d/%d dist=%.2f speed_kmh=%.1f",
                int(self._route_parking_debug_step_counter),
                int(self._route_parking_index),
                len(self._route_parking_points),
                float(dist_to_target),
                float(speed_kmh),
            )

        if dist_to_target <= self._route_parking_arrival_distance_m:
            self._route_parking_index += 1
            LOGGER.info(
                "[route] parking-step: reached waypoint, advance_to_index=%d/%d",
                self._route_parking_index,
                len(self._route_parking_points),
            )
            if self._route_parking_index >= len(self._route_parking_points):
                stop_control = carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    steer=0.0,
                    hand_brake=True
                )
                self.vehicle.apply_control(stop_control)

                self._route_status = "arrived"
                self._route_active = False
                self._route_parking_active = False
                LOGGER.info("[route] parking-step: final parking waypoint reached; vehicle stopped")
                return self._route_status
            target = self._route_parking_points[self._route_parking_index]

        transform = self.vehicle.get_transform()
        yaw_rad = math.radians(float(transform.rotation.yaw))
        heading_x = math.cos(yaw_rad)
        heading_y = math.sin(yaw_rad)
        vec_x = float(target[0]) - float(transform.location.x)
        vec_y = float(target[1]) - float(transform.location.y)
        vec_norm = math.sqrt(vec_x * vec_x + vec_y * vec_y)

        if vec_norm < 1e-4:
            steer = 0.0
        else:
            target_x = vec_x / vec_norm
            target_y = vec_y / vec_norm
            cross = heading_x * target_y - heading_y * target_x
            steer = max(-0.6, min(0.6, 1.35 * cross))

        speed_kmh = self.get_speed_kmh()
        target_speed = self._route_parking_target_speed_kmh
        if speed_kmh < target_speed:
            throttle = min(0.26, 0.08 + 0.03 * (target_speed - speed_kmh))
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(0.35, 0.06 * (speed_kmh - target_speed))

        control = carla.VehicleControl(
            throttle=float(max(0.0, min(1.0, throttle))),
            brake=float(max(0.0, min(1.0, brake))),
            steer=float(steer),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )
        self.vehicle.apply_control(control)
        self._route_status = "parking-running"
        return self._route_status

    def _distance_to_route_m(self, current_location) -> float | None:
        if not self._route_polyline:
            return None
        min_distance = float("inf")
        for point in self._route_polyline:
            dx = float(current_location.x) - float(point[0])
            dy = float(current_location.y) - float(point[1])
            dz = float(current_location.z) - float(point[2])
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            if distance < min_distance:
                min_distance = distance
        return None if min_distance == float("inf") else min_distance

    def run_route_step(self) -> str:
        if carla is None or self.vehicle is None:
            self._route_status = "failed"
            return self._route_status
        if not self._route_active:
            return self._route_status
        if self._route_destination is None:
            self._route_active = False
            self._route_status = "failed"
            return self._route_status

        if self._route_parking_active:
            LOGGER.debug(
                "[route] step-mode=parking index=%d/%d status=%s",
                self._route_parking_index,
                len(self._route_parking_points),
                self._route_status,
            )
            return self._run_parking_waypoint_step()

        if self._route_agent is None:
            self._route_active = False
            self._route_status = "failed"
            return self._route_status

        self._route_debug_step_counter += 1
        if self._route_debug_step_counter % 20 == 0:
            LOGGER.info(
                "[route] step-mode=normal tick=%d parking_active=%s parking_points=%d status=%s",
                self._route_debug_step_counter,
                self._route_parking_active,
                len(self._route_parking_points),
                self._route_status,
            )

        destination_location = carla.Location(*self._route_destination)
        current_location = self.vehicle.get_location()
        if current_location.distance(destination_location) <= self._route_arrival_distance_m:
            stop_control = self._build_safe_stop_control()
            if stop_control is not None:
                try:
                    self.vehicle.apply_control(stop_control)
                except Exception:
                    pass
            self._route_active = False
            self._route_agent = None
            self._route_parking_active = False
            self._route_parking_index = 0
            self._route_parking_points = []
            self._route_status = "arrived"
            self._route_offtrack_counter = 0
            return self._route_status

        if self._route_reroute_cooldown_ticks > 0:
            self._route_reroute_cooldown_ticks -= 1

        distance_to_route = self._distance_to_route_m(current_location)
        if distance_to_route is not None and distance_to_route > self._route_reroute_distance_threshold_m:
            self._route_offtrack_counter += 1
        else:
            self._route_offtrack_counter = 0

        if (
            self._route_offtrack_counter >= self._route_reroute_required_ticks
            and self._route_reroute_cooldown_ticks <= 0
        ):
            replanned, _ = self.plan_route_to_destination()
            if replanned and BasicAgent is not None:
                try:
                    destination_location = carla.Location(*self._route_destination)
                    self._route_agent = BasicAgent(self.vehicle)
                    if hasattr(self._route_agent, "set_target_speed"):
                        self._route_agent.set_target_speed(30.0)
                    if hasattr(self._route_agent, "set_destination"):
                        self._route_agent.set_destination(destination_location)
                    self._route_status = "rerouted"
                except Exception:
                    self._route_active = False
                    self._route_agent = None
                    self._route_status = "failed"
                    return self._route_status
            self._route_offtrack_counter = 0
            self._route_reroute_cooldown_ticks = self._route_reroute_cooldown_default_ticks

        # ADD THIS BLOCK just before the final try/except that calls run_step()
        # ── Parking handoff: switch controller when vehicle is near parking entry ──
        if (
            not self._route_parking_active
            and self._route_parking_points
            and self._route_agent is not None
        ):
            parking_entry = self._route_parking_points[0]
            dist_to_entry = self._distance_xyz(
                (float(current_location.x), float(current_location.y), float(current_location.z)),
                parking_entry,
            )
            if dist_to_entry <= 6.0:   # hand off when within 6 m of parking entry
                try:
                    speed_kmh = self.get_speed_kmh()
                except Exception:
                    speed_kmh = 0.0
                LOGGER.info(
                    "[route] handoff: switching to parking controller dist_to_entry=%.2f speed_kmh=%.1f entry=(%.2f, %.2f, %.2f)",
                    float(dist_to_entry),
                    float(speed_kmh),
                    float(parking_entry[0]),
                    float(parking_entry[1]),
                    float(parking_entry[2]),
                )
                self._route_parking_active = True
                self._route_parking_index = 0
                self._route_parking_debug_step_counter = 0
                self._route_agent = None        # release BasicAgent
                return self._run_parking_waypoint_step()
        # ──────────────────────────────────────────────────────────────────────────────

        try:
            control = self._route_agent.run_step()
            control.manual_gear_shift = False
            control.hand_brake = False
            self.vehicle.apply_control(control)
            return self._route_status
        except Exception:
            self._route_active = False
            self._route_agent = None
            self._route_status = "failed"
            return self._route_status

# REPLACE the entire _collect_parking_shoulder_waypoints method

    def _collect_parking_shoulder_waypoints(
        self,
        goal_hint: tuple[float, float, float],
        search_radius_m: float = 80.0,
    ) -> list[tuple[float, float, float]]:
        """Collect Parking/Shoulder waypoints by walking the full lane length.

        generate_waypoints() only yields driving-lane waypoints. We reach
        parking/shoulder lanes via get_right_lane(), then walk the entire
        lane forward and backward from each seed to get full coverage.
        """
        if carla is None or self.world is None:
            return []

        world_map = self.world.get_map()
        goal_location = carla.Location(*goal_hint)

        # 1. Sample driving waypoints near the goal area
        all_driving = world_map.generate_waypoints(2.0)
        nearby_driving = [
            wp for wp in all_driving
            if wp.transform.location.distance(goal_location) <= search_radius_m
        ]

        seen_keys: set[tuple[float, float, float]] = set()
        candidates: list[tuple[float, float, float]] = []

        def _key(wp) -> tuple[float, float, float]:
            loc = wp.transform.location
            return (round(float(loc.x), 1), round(float(loc.y), 1), round(float(loc.z), 1))

        def _add(wp) -> bool:
            """Add waypoint if unseen. Returns True if it was new."""
            k = _key(wp)
            if k in seen_keys:
                return False
            seen_keys.add(k)
            loc = wp.transform.location
            candidates.append((float(loc.x), float(loc.y), float(loc.z)))
            return True

        def _walk_full_lane(seed_wp, max_steps: int = 200) -> None:
            """Walk forward and backward from seed along parking/shoulder lane."""
            # Walk forward
            cursor = seed_wp
            for _ in range(max_steps):
                if not _add(cursor):
                    break
                nexts = cursor.next(2.0)
                if not nexts:
                    break
                # Stay on same lane type
                same = [w for w in nexts if w.lane_type == seed_wp.lane_type]
                cursor = same[0] if same else nexts[0]

            # Walk backward
            cursor = seed_wp
            for _ in range(max_steps):
                prevs = cursor.previous(2.0)
                if not prevs:
                    break
                same = [w for w in prevs if w.lane_type == seed_wp.lane_type]
                cursor = same[0] if same else prevs[0]
                if not _add(cursor):
                    break

        lane_seeds_seen: set[tuple[int, int, int]] = set()

        for driving_wp in nearby_driving:
            # Walk rightward to find parking/shoulder neighbors
            cursor = driving_wp.get_right_lane()
            depth = 0
            while cursor is not None and depth < 6:
                if cursor.lane_type in (carla.LaneType.Parking, carla.LaneType.Shoulder):
                    # Use (road_id, section_id, lane_id) to avoid re-walking same lane
                    lane_key = (
                        int(cursor.road_id),
                        int(cursor.section_id),
                        int(cursor.lane_id),
                    )
                    if lane_key not in lane_seeds_seen:
                        lane_seeds_seen.add(lane_key)
                        _walk_full_lane(cursor)   # ← walk the WHOLE lane, not just 1 step
                cursor = cursor.get_right_lane()
                depth += 1

        LOGGER.info(
            "[route] _collect_parking_shoulder_waypoints: found=%d lanes_walked=%d radius=%.1f",
            len(candidates),
            len(lane_seeds_seen),
            float(search_radius_m),
        )
        return candidates

    def get_route_destination(self) -> tuple[float, float, float] | None:
        return None if self._route_destination is None else tuple(self._route_destination)

    def get_route_polyline(self) -> list[tuple[float, float, float]]:
        return [tuple(point) for point in self._route_polyline]

    def get_route_status(self) -> str:
        return self._route_status

    def has_route_plan(self) -> bool:
        return self._route_planned

    def is_route_following(self) -> bool:
        return self._route_active

    def engage_sleep_safety_autopilot_right_lane(self) -> tuple[bool, str]:
        if carla is None:
            return False, "CARLA API unavailable"
        if self.vehicle is None:
            return False, "No CARLA vehicle is attached"

        if not self.set_autopilot_enabled(True):
            return False, "Failed to enable autopilot"

        messages: list[str] = ["autopilot enabled"]
        waypoint = self._get_current_waypoint_for_parking()
        can_change_right = False
        if waypoint is not None:
            right_lane = waypoint.get_right_lane()
            lane_change = waypoint.lane_change
            can_change_right = right_lane is not None and bool(lane_change & carla.LaneChange.Right)

        if self._traffic_manager is not None:
            try:
                self._traffic_manager.auto_lane_change(self.vehicle, False)
                messages.append("automatic lane changes disabled")
            except Exception:
                messages.append("could not lock lane-change policy")

            if can_change_right:
                try:
                    self._traffic_manager.force_lane_change(self.vehicle, True)
                    messages.append("requested immediate right-lane change")
                except Exception:
                    messages.append("right-lane change request failed")
            else:
                messages.append("already rightmost or no legal right-lane change")
        else:
            if can_change_right:
                messages.append("traffic manager unavailable; cannot force right-lane change")
            else:
                messages.append("already rightmost or no legal right-lane change")

        return True, "; ".join(messages)

    def get_speed_kmh(self) -> float:
        if self.vehicle is None:
            return 0.0
        velocity = self.vehicle.get_velocity()
        speed_mps = math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z)
        return speed_mps * 3.6

    def apply_drowsy_slowdown(self, target_speed_kmh: float = 15.0, brake_strength: float = 0.35) -> bool:
        if carla is None or self.vehicle is None:
            return False

        current_speed = self.get_speed_kmh()
        if current_speed <= target_speed_kmh:
            return False

        control = self.vehicle.get_control()
        control.throttle = min(control.throttle, 0.15)
        control.brake = max(control.brake, max(0.1, min(1.0, brake_strength)))
        control.hand_brake = False
        self.vehicle.apply_control(control)
        return True

    def start_autonomous_roadside_parking(self) -> bool:
        self._parking_active = False
        self._parking_completed = False
        self._parking_mode = "disabled"
        self._parking_phase = "disabled"
        self._parking_agent = None
        self._parking_last_error = "Autonomous parking logic is disabled"
        return False

    def run_autonomous_parking_step(self, top_gap_px: int | None = None, bottom_gap_px: int | None = None) -> str:
        del top_gap_px, bottom_gap_px
        self._parking_active = False
        self._parking_mode = "disabled"
        self._parking_phase = "disabled"
        self._parking_agent = None
        self._parking_last_error = "Autonomous parking logic is disabled"
        return "disabled"

    def _get_current_waypoint_for_parking(self):
        if carla is None or self.world is None or self.vehicle is None:
            return None
        return self.world.get_map().get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Parking,
        )

    def is_in_shoulder_or_parking_lane(self) -> bool:
        """Return True if the vehicle is currently on a shoulder or parking lane."""
        if carla is None or self.world is None or self.vehicle is None:
            return False

        waypoint = self._get_current_waypoint_for_parking()
        if waypoint is None:
            return False

        try:
            lane_type = waypoint.lane_type
        except Exception:
            return False

        return bool(lane_type & (carla.LaneType.Shoulder | carla.LaneType.Parking))

    def is_in_rightmost_driving_lane(self) -> bool:
        """Return True if the vehicle is already in the rightmost driving lane.

        If the vehicle is on a shoulder/parking lane, treat it as already rightmost.
        """
        if carla is None or self.world is None or self.vehicle is None:
            return True

        try:
            waypoint = self.world.get_map().get_waypoint(
                self.vehicle.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Parking,
            )
        except Exception:
            waypoint = None

        if waypoint is None:
            return True

        try:
            current_type = waypoint.lane_type
        except Exception:
            return True

        if bool(current_type & (carla.LaneType.Shoulder | carla.LaneType.Parking)):
            return True

        try:
            right_neighbor = waypoint.get_right_lane()
        except Exception:
            right_neighbor = None

        if right_neighbor is None:
            return True

        try:
            right_type = right_neighbor.lane_type
        except Exception:
            return True

        # If there's another DRIVING lane to the right, we're not rightmost yet.
        return not bool(right_type & carla.LaneType.Driving)

    def _has_right_shoulder_or_parking_neighbor(self, waypoint) -> bool:
        if carla is None or waypoint is None:
            return False
        current_type = waypoint.lane_type
        if current_type & (carla.LaneType.Shoulder | carla.LaneType.Parking):
            return True
        right_neighbor = waypoint.get_right_lane()
        if right_neighbor is None:
            return False
        right_type = right_neighbor.lane_type
        return bool(right_type & (carla.LaneType.Shoulder | carla.LaneType.Parking))

    def is_positioned_for_right_shoulder_parking(self) -> bool:
        """Return True when the vehicle is in a driving lane with shoulder/parking on the right.

        Sleepy auto-park uses this as a precondition so the eventual move into the
        shoulder/parking lane is gentle (no sharp lateral cut from a left lane).

        If the vehicle is already on a shoulder/parking lane, this returns True.
        """
        if carla is None or self.world is None or self.vehicle is None:
            return True

        waypoint = self._get_current_waypoint_for_parking()
        if waypoint is None:
            return True

        try:
            return self._has_right_shoulder_or_parking_neighbor(waypoint)
        except Exception:
            return True

    def _can_change_to_right_lane(self, waypoint) -> bool:
        if carla is None or waypoint is None:
            return False
        if waypoint.get_right_lane() is None:
            return False
        lane_change = waypoint.lane_change
        return bool(lane_change & carla.LaneChange.Right)

    def _lane_lateral_error_m(self, waypoint) -> float:
        if self.vehicle is None:
            return 0.0
        vehicle_location = self.vehicle.get_location()
        lane_center = waypoint.transform.location
        delta_x = vehicle_location.x - lane_center.x
        delta_y = vehicle_location.y - lane_center.y
        yaw_rad = math.radians(waypoint.transform.rotation.yaw)
        right_x = math.cos(yaw_rad + math.pi / 2.0)
        right_y = math.sin(yaw_rad + math.pi / 2.0)
        return delta_x * right_x + delta_y * right_y

    def _waypoint_lane_key(self, waypoint) -> tuple[int, int, int] | None:
        if waypoint is None:
            return None
        return (int(waypoint.road_id), int(waypoint.section_id), int(waypoint.lane_id))

    def _has_changed_from_start_lane(self, waypoint) -> bool:
        current_key = self._waypoint_lane_key(waypoint)
        if current_key is None or self._parking_lane_change_start_key is None:
            return False
        return current_key != self._parking_lane_change_start_key

    def _is_effectively_rightmost_after_positioning(self, waypoint) -> bool:
        if waypoint is None:
            return False
        if self._parking_positioning_ticks < self._parking_min_positioning_ticks_before_rightmost_check:
            return False
        # Some CARLA maps keep lane identifiers stable during lateral movement,
        # so also treat "cannot change further right" as lane-change completion.
        return not self._can_change_to_right_lane(waypoint)

    def _is_waypoint_in_junction(self, waypoint) -> bool:
        if waypoint is None:
            return False
        try:
            return bool(waypoint.is_junction)
        except Exception:
            return False

    def _run_junction_basic_agent(self, waypoint) -> bool:
        if carla is None or self.vehicle is None or waypoint is None or BasicAgent is None:
            return False

        destination = self._find_junction_exit_destination(waypoint)
        if destination is None:
            return False

        if self._parking_agent is None:
            try:
                self._parking_agent = BasicAgent(self.vehicle)
            except Exception:
                self._parking_agent = None
                return False

        try:
            if hasattr(self._parking_agent, "set_target_speed"):
                self._parking_agent.set_target_speed(self._parking_junction_agent_speed_kmh)
            if hasattr(self._parking_agent, "set_destination"):
                self._parking_agent.set_destination(destination)
            agent_control = self._parking_agent.run_step()
            agent_control.manual_gear_shift = False
            agent_control.hand_brake = False
            agent_control.reverse = False
            agent_control.steer = max(-0.3, min(0.3, getattr(agent_control, "steer", 0.0)))
            self.vehicle.apply_control(agent_control)
            return True
        except Exception:
            self._parking_agent = None
            return False

    def _find_junction_exit_destination(self, waypoint):
        if carla is None or waypoint is None:
            return None

        frontier = [waypoint]
        seen: set[tuple[float, float, float]] = set()
        search_depth = 0

        while frontier and search_depth < 20:
            current = frontier.pop(0)
            search_depth += 1

            location = current.transform.location
            location_key = (round(location.x, 1), round(location.y, 1), round(location.z, 1))
            if location_key in seen:
                continue
            seen.add(location_key)

            if not self._is_waypoint_in_junction(current):
                return location

            next_waypoints = current.next(5.0)
            for candidate in next_waypoints:
                candidate_location = candidate.transform.location
                candidate_key = (round(candidate_location.x, 1), round(candidate_location.y, 1), round(candidate_location.z, 1))
                if candidate_key not in seen:
                    frontier.append(candidate)

        return None

    def _parking_speed_command(self, target_speed_kmh: float, max_throttle: float = 0.07, max_brake: float = 0.22) -> tuple[float, float]:
        current_speed_kmh = self.get_speed_kmh()
        speed_error = target_speed_kmh - current_speed_kmh

        if speed_error >= 0:
            throttle = min(max_throttle, max(0.0, speed_error * 0.08))
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(max_brake, max(0.0, (-speed_error) * 0.12))

        if current_speed_kmh < 0.5 and target_speed_kmh <= 1.2:
            throttle = min(throttle, 0.03)

        return throttle, brake

    @property
    def parking_active(self) -> bool:
        return self._parking_active

    @property
    def parking_completed(self) -> bool:
        return self._parking_completed

    @property
    def parking_phase(self) -> str:
        return self._parking_phase

    @property
    def parking_last_error(self) -> str:
        return self._parking_last_error

    def _find_roadside_target_location(self):
        if carla is None or self.world is None or self.vehicle is None:
            return None

        current_wp = self.world.get_map().get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Parking,
        )
        if current_wp is None:
            return None

        right_wp = current_wp
        for _ in range(4):
            candidate = right_wp.get_right_lane()
            if candidate is None:
                break
            right_wp = candidate
            if right_wp.lane_type in (carla.LaneType.Shoulder, carla.LaneType.Parking):
                break

        next_points = right_wp.next(18.0)
        target_wp = next_points[0] if next_points else right_wp
        return target_wp.transform.location

    def _build_safe_stop_control(self):
        if carla is None:
            return None
        return carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0, hand_brake=True)

    def get_all_map_waypoints(
        self,
        resolution: float = 2.0,
        include_junctions: bool = True,
        max_waypoints: int | None = None,
    ) -> list[dict]:
        """Return a sampled list of waypoints for the loaded CARLA map.

        The result is cached per map and sampling configuration so the dashboard
        can refresh a map view without rebuilding the full waypoint cloud every tick.
        """
        if carla is None or self.world is None:
            return []

        try:
            world_map = self.world.get_map()
            map_name = getattr(world_map, "name", "") or "unknown_map"
            cache_key = (map_name, float(resolution), bool(include_junctions), max_waypoints)
            if self._map_waypoint_cache_key == cache_key and self._map_waypoint_cache:
                return [dict(waypoint) for waypoint in self._map_waypoint_cache]

            sampled_waypoints = world_map.generate_waypoints(float(max(0.1, resolution)))
            collected: list[dict] = []
            for waypoint in sampled_waypoints:
                try:
                    is_junction = bool(waypoint.is_junction)
                except Exception:
                    is_junction = False

                if not include_junctions and is_junction:
                    continue

                location = waypoint.transform.location
                is_rightmost_lane = False
                try:
                    if waypoint.lane_type == carla.LaneType.Driving:
                        right_lane = waypoint.get_right_lane()
                        has_driving_right_lane = False
                        while right_lane is not None:
                            if right_lane.lane_type == carla.LaneType.Driving:
                                has_driving_right_lane = True
                                break
                            right_lane = right_lane.get_right_lane()
                        is_rightmost_lane = not has_driving_right_lane
                except Exception:
                    is_rightmost_lane = False

                collected.append(
                    {
                        "location": (float(location.x), float(location.y), float(location.z)),
                        "road_id": int(waypoint.road_id),
                        "section_id": int(waypoint.section_id),
                        "lane_id": int(waypoint.lane_id),
                        "lane_type": waypoint.lane_type,
                        "is_junction": is_junction,
                        "is_rightmost_lane": is_rightmost_lane,
                    }
                )

                if is_rightmost_lane and not is_junction:
                    try:
                        right_vec = waypoint.transform.get_right_vector()
                        offset_x = float(location.x) + float(right_vec.x) * 3.30
                        offset_y = float(location.y) + float(right_vec.y) * 3.30
                        offset_z = float(location.z) + float(right_vec.z) * 3.30
                    except Exception:
                        offset_x = float(location.x)
                        offset_y = float(location.y)
                        offset_z = float(location.z)

                    collected.append(
                        {
                            "location": (offset_x, offset_y, offset_z),
                            "road_id": int(waypoint.road_id),
                            "section_id": int(waypoint.section_id),
                            "lane_id": int(waypoint.lane_id),
                            "lane_type": waypoint.lane_type,
                            "is_junction": is_junction,
                            "is_rightmost_lane": False,
                            "is_right_offset": True,
                        }
                    )

                if max_waypoints is not None and len(collected) >= max_waypoints:
                    break

            self._map_waypoint_cache_key = cache_key
            self._map_waypoint_cache = collected
            return [dict(waypoint) for waypoint in collected]
        except Exception:
            return []

    def get_waypoints_ahead(self, distance_ahead_meters: float = 100.0, max_waypoints: int = 20) -> list[dict]:
        """Get upcoming waypoints ahead of the vehicle on the current route.

        Args:
            distance_ahead_meters: How far ahead to look for waypoints (default 100m)
            max_waypoints: Maximum number of waypoints to return (default 20)

        Returns:
            List of dicts with keys: 'distance', 'location', 'road_id', 'lane_id', 'lane_type'
        """
        if carla is None or self.world is None or self.vehicle is None:
            return []

        try:
            current_location = self.vehicle.get_location()
            current_waypoint = self.world.get_map().get_waypoint(
                current_location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Parking,
            )

            if current_waypoint is None:
                return []

            waypoints = []
            search_distance = 0.0
            frontier = [current_waypoint]
            visited = set()

            while frontier and len(waypoints) < max_waypoints and search_distance < distance_ahead_meters:
                current_wp = frontier.pop(0)

                # Create a unique key to avoid revisiting waypoints
                wp_key = (current_wp.road_id, current_wp.section_id, current_wp.lane_id)
                if wp_key in visited:
                    continue
                visited.add(wp_key)

                wp_location = current_wp.transform.location
                distance_from_vehicle = current_location.distance(wp_location)

                # Only include waypoints that are ahead of the vehicle
                if distance_from_vehicle > 5.0:  # Skip very close waypoints (current position)
                    waypoints.append({
                        "distance": distance_from_vehicle,
                        "location": (wp_location.x, wp_location.y, wp_location.z),
                        "road_id": current_wp.road_id,
                        "lane_id": current_wp.lane_id,
                        "lane_type": current_wp.lane_type,
                    })

                # Get next waypoints (5 meters apart)
                if search_distance < distance_ahead_meters:
                    next_wps = current_wp.next(5.0)
                    for next_wp in next_wps:
                        next_key = (next_wp.road_id, next_wp.section_id, next_wp.lane_id)
                        if next_key not in visited:
                            frontier.append(next_wp)
                    search_distance += 5.0

            # Sort by distance
            waypoints.sort(key=lambda w: w["distance"])
            return waypoints

        except Exception:
            return []

    def _world_to_camera_2d(self, world_point: tuple[float, float, float]) -> tuple[int, int] | None:
        """Project a 3D world point to 2D camera image coordinates.

        Args:
            world_point: (x, y, z) in CARLA world coordinates

        Returns:
            (pixel_x, pixel_y) or None if point is behind camera or not in view
        """
        if carla is None or self.vehicle is None or self.camera_sensor is None:
            return None

        try:
            # Get camera transform relative to vehicle
            camera_transform = self.camera_sensor.get_transform()
            vehicle_transform = self.vehicle.get_transform()

            # Convert world point to camera-relative coordinates
            camera_location = camera_transform.location
            camera_rotation = camera_transform.rotation

            # Translate world point to camera origin
            point_relative = carla.Vector3D(
                world_point[0] - camera_location.x,
                world_point[1] - camera_location.y,
                world_point[2] - camera_location.z,
            )

            # Rotate to camera frame (convert to camera's coordinate system)
            yaw = math.radians(camera_rotation.yaw)
            pitch = math.radians(camera_rotation.pitch)
            roll = math.radians(camera_rotation.roll)

            # Apply rotation matrices
            # First rotate around Z (yaw)
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            temp_x = point_relative.x * cos_yaw - point_relative.y * sin_yaw
            temp_y = point_relative.x * sin_yaw + point_relative.y * cos_yaw
            temp_z = point_relative.z

            # Then rotate around Y (pitch)
            cos_pitch = math.cos(pitch)
            sin_pitch = math.sin(pitch)
            cam_x = temp_x * cos_pitch + temp_z * sin_pitch
            cam_y = temp_y
            cam_z = -temp_x * sin_pitch + temp_z * cos_pitch

            # Point must be in front of camera (positive Z)
            if cam_z <= 0.1:
                return None

            # Project to 2D using camera intrinsics
            # Standard CARLA camera: f = width / (2 * tan(fov/2))
            fov_rad = math.radians(self._camera_fov)
            focal_length = self._camera_width / (2.0 * math.tan(fov_rad / 2.0))

            # Principal point (typically image center)
            cx = self._camera_width / 2.0
            cy = self._camera_height / 2.0

            # Project 3D to 2D
            x_2d = int((cam_x / cam_z) * focal_length + cx)
            y_2d = int((cam_y / cam_z) * focal_length + cy)

            # Check if point is within image bounds
            if 0 <= x_2d < self._camera_width and 0 <= y_2d < self._camera_height:
                return (x_2d, y_2d)

            return None

        except Exception:
            return None

    def draw_waypoints_on_frame(self, frame: np.ndarray | None, waypoints: list[dict]) -> np.ndarray | None:
        """Draw waypoint path overlay on CARLA camera frame.

        Args:
            frame: RGB frame from CARLA camera (or None)
            waypoints: List of waypoint dicts with 'location' and 'distance' keys

        Returns:
            Annotated frame with waypoint path drawn, or input frame if not available
        """
        if frame is None or len(waypoints) == 0:
            return frame

        try:
            import cv2  # type: ignore
        except Exception:
            return frame

        annotated = frame.copy()
        projected_points = []

        # Project all waypoints to 2D
        for wp in waypoints:
            point_2d = self._world_to_camera_2d(wp["location"])
            if point_2d is not None:
                projected_points.append(point_2d)

        # Draw path line connecting waypoints
        if len(projected_points) >= 2:
            pts = np.array(projected_points, dtype=np.int32)
            cv2.polylines(annotated, [pts], False, (0, 255, 0), 2)  # Green path line

        # Draw waypoint markers and distance labels
        proj_idx = 0
        for wp in waypoints:
            point_2d = self._world_to_camera_2d(wp["location"])
            if point_2d is not None:
                # Draw circle at waypoint
                cv2.circle(annotated, point_2d, 6, (0, 255, 0), -1)  # Green filled circle
                cv2.circle(annotated, point_2d, 6, (255, 255, 0), 1)  # Cyan outline

                # Draw distance label
                distance_text = f"{wp['distance']:.1f}m"
                text_pos = (point_2d[0] + 10, point_2d[1] - 5)
                cv2.putText(
                    annotated,
                    distance_text,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                proj_idx += 1

        return annotated

