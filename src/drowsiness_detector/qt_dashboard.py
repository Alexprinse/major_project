from __future__ import annotations

import logging
import sys
import time
import os
import re
import math
import subprocess
import signal
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import cv2  # type: ignore
import numpy as np


LOGGER = logging.getLogger(__name__)

try:
    import carla  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    carla = None

try:
    import pyqtgraph as pg  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency at runtime
    pg = None

try:
    from PyQt5.QtCore import QLibraryInfo, QTimer, Qt, QRect  # type: ignore[import-not-found]
    from PyQt5.QtGui import QImage, QPixmap, QColor, QRegion  # type: ignore[import-not-found]
    from PyQt5.QtWidgets import (  # type: ignore[import-not-found]
        QApplication,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QPlainTextEdit,
        QTextEdit,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
except Exception:  # pragma: no cover - optional dependency at runtime
    QLibraryInfo = None
    QApplication = None
    QTimer = None
    Qt = None
    QRect = None
    QFrame = None
    QGridLayout = None
    QHBoxLayout = None
    QImage = None
    QPixmap = None
    QColor = None
    QRegion = None
    QLabel = None
    QMainWindow = object
    QMessageBox = None
    QProgressBar = None
    QPushButton = None
    QPlainTextEdit = None
    QTextEdit = None
    QTableWidget = None
    QTableWidgetItem = None
    QTabWidget = None
    QVBoxLayout = None
    QWidget = None

try:
    from .alerts import AlertController
    from .carla_controller import CarlaSafetyController
    from .detector import DrowsinessDetector, DrowsinessState
except ImportError:
    # Allow direct execution: python src/drowsiness_detector/qt_dashboard.py
    repo_src = Path(__file__).resolve().parents[1]
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    from drowsiness_detector.alerts import AlertController
    from drowsiness_detector.carla_controller import CarlaSafetyController
    from drowsiness_detector.detector import DrowsinessDetector, DrowsinessState


_QT_APP = None


def _find_weather_presets() -> list[tuple[object, str]]:
    if carla is None:
        return []
    rgx = re.compile(r".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")
    presets = [x for x in dir(carla.WeatherParameters) if re.match("[A-Z].+", x)]
    return [(getattr(carla.WeatherParameters, x), " ".join(m.group(0) for m in rgx.finditer(x))) for x in presets]


def _lane_type_name(lane_type: object) -> str:
    text = str(lane_type)
    return text.split(".")[-1] if "." in text else text


def _get_carla_lane_info(controller: CarlaSafetyController) -> tuple[str, str, str, str]:
    if carla is None or controller.world is None or controller.vehicle is None:
        return "Lane: N/A", "Road/Section: N/A", "Neighbors: L=N/A R=N/A", "Lane change: N/A"

    try:
        world_map = controller.world.get_map()
        waypoint = world_map.get_waypoint(
            controller.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Parking,
        )
        if waypoint is None:
            return "Lane: N/A", "Road/Section: N/A", "Neighbors: L=N/A R=N/A", "Lane change: N/A"

        lane_type = _lane_type_name(waypoint.lane_type)
        lane_line = f"Lane: id={waypoint.lane_id} type={lane_type}"
        road_line = f"Road/Section: road={waypoint.road_id} section={waypoint.section_id}"

        left_wp = waypoint.get_left_lane()
        right_wp = waypoint.get_right_lane()
        left_type = _lane_type_name(left_wp.lane_type) if left_wp is not None else "None"
        right_type = _lane_type_name(right_wp.lane_type) if right_wp is not None else "None"
        neighbors_line = f"Neighbors: L={left_type} R={right_type}"

        lane_change_line = f"Lane change: {_lane_type_name(waypoint.lane_change)}"
        return lane_line, road_line, neighbors_line, lane_change_line
    except Exception:
        return "Lane: N/A", "Road/Section: N/A", "Neighbors: L=N/A R=N/A", "Lane change: N/A"


def _to_qpixmap_bgr(frame_bgr):
    if QImage is None or QPixmap is None:
        return None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def _build_placeholder_frame(width: int, height: int, title: str, subtitle: str = ""):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (8, 14, 24)
    cv2.rectangle(frame, (8, 8), (width - 8, height - 8), (48, 74, 110), 2)
    cv2.putText(frame, title, (24, max(42, height // 2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 242, 251), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(frame, subtitle, (24, min(height - 28, height // 2 + 26)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (157, 193, 230), 1, cv2.LINE_AA)
    return frame


def _build_minimap_frame(
    map_waypoints: list[dict],
    vehicle_location,
    route_points: list[tuple[float, float, float]] | None,
    destination: tuple[float, float, float] | None,
    heading_deg: float,
    size: int = 220,
    meters_radius: float = 60.0,
) -> np.ndarray:
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:] = (14, 20, 30)
    center = size // 2

    cv2.circle(canvas, (center, center), int(size * 0.46), (34, 50, 72), 1)
    cv2.circle(canvas, (center, center), int(size * 0.32), (28, 42, 62), 1)

    if vehicle_location is None:
        cv2.putText(canvas, "NO GPS", (center - 36, center + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 205, 224), 1, cv2.LINE_AA)
        return canvas

    yaw_rad = math.radians(float(heading_deg))
    forward = (math.cos(yaw_rad), math.sin(yaw_rad))
    right = (math.cos(yaw_rad + math.pi / 2.0), math.sin(yaw_rad + math.pi / 2.0))
    scale = (size * 0.46) / max(1.0, float(meters_radius))

    def project(world_x: float, world_y: float) -> tuple[int, int] | None:
        dx = float(world_x) - float(vehicle_location.x)
        dy = float(world_y) - float(vehicle_location.y)
        proj_right = dx * right[0] + dy * right[1]
        proj_forward = dx * forward[0] + dy * forward[1]
        px = int(center + proj_right * scale)
        py = int(center - proj_forward * scale)
        if px < 0 or px >= size or py < 0 or py >= size:
            return None
        return px, py

    for wp in map_waypoints:
        loc = wp.get("location")
        if loc is None:
            continue
        point = project(loc[0], loc[1])
        if point is not None:
            cv2.circle(canvas, point, 1, (74, 126, 178), -1)

    if route_points and len(route_points) >= 2:
        closest_idx = 0
        closest_dist_sq = float("inf")
        ego_x = float(vehicle_location.x)
        ego_y = float(vehicle_location.y)
        for idx, point in enumerate(route_points):
            dx = float(point[0]) - ego_x
            dy = float(point[1]) - ego_y
            dist_sq = dx * dx + dy * dy
            if dist_sq < closest_dist_sq:
                closest_dist_sq = dist_sq
                closest_idx = idx

        cleared_pixels: list[tuple[int, int]] = []
        for point in route_points[: closest_idx + 1]:
            mapped = project(point[0], point[1])
            if mapped is not None:
                cleared_pixels.append(mapped)

        remaining_pixels: list[tuple[int, int]] = []
        for point in route_points[closest_idx:]:
            mapped = project(point[0], point[1])
            if mapped is not None:
                remaining_pixels.append(mapped)

        if len(cleared_pixels) >= 2:
            cv2.polylines(canvas, [np.array(cleared_pixels, dtype=np.int32)], False, (58, 74, 92), 2)
        if len(remaining_pixels) >= 2:
            cv2.polylines(canvas, [np.array(remaining_pixels, dtype=np.int32)], False, (127, 209, 79), 2)

    if destination is not None:
        destination_point = project(destination[0], destination[1])
        if destination_point is not None:
            cv2.circle(canvas, destination_point, 6, (71, 48, 175), -1)
            cv2.circle(canvas, destination_point, 8, (93, 64, 220), 1)

    vehicle_triangle = np.array(
        [
            (center, center - 12),
            (center - 8, center + 8),
            (center + 8, center + 8),
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(canvas, vehicle_triangle, (86, 196, 255))
    cv2.polylines(canvas, [vehicle_triangle], True, (214, 241, 255), 1)
    cv2.circle(canvas, (center, center), int(size * 0.46), (58, 86, 118), 2)
    return canvas


def _state_text(state: DrowsinessState) -> str:
    value = state.value.upper()
    return "SLEEP" if value == "SLEEPY" else value


def _badge_style(kind: str) -> str:
    palette = {
        "good": "#173d2a|#2ea769|#eafff3",
        "warn": "#4a3d12|#c99b28|#fff5d6",
        "bad": "#5f1f2d|#a43a57|#ffe9ef",
        "info": "#163252|#2d80c6|#eef7ff",
        "neutral": "#1a2230|#33445a|#e5eef8",
    }
    background, border, color = palette.get(kind, palette["neutral"]).split("|")
    return (
        f"background: {background}; border: 1px solid {border}; border-radius: 10px; "
        f"padding: 6px 12px; color: {color}; font-weight: 700;"
    )


def _set_badge(label, text: str, kind: str = "neutral") -> None:
    label.setText(text)
    label.setStyleSheet(_badge_style(kind))


@dataclass(slots=True)
class DashboardSnapshot:
    driver_state: str = "ALERT"
    mode: str = "MANUAL"
    parking_state: str = "Manual"
    speed_kmh: float = 0.0
    client_fps: float = 0.0
    server_fps: float = 0.0
    ear: float = 0.0
    drowsiness_score: int = 0
    gear: int = 1
    reverse_enabled: bool = False
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    camera_label: str = "Camera"
    camera_mode: str = "RGB"
    lane_hint: str = "Lane hint: N/A"
    alert_lines: list[str] = None  # type: ignore[assignment]
    decision_lines: list[str] = None  # type: ignore[assignment]
    telemetry_lines: list[str] = None  # type: ignore[assignment]
    trajectory_points: list[tuple[float, float]] = None  # type: ignore[assignment]
    waypoints: list[dict] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.alert_lines is None:
            self.alert_lines = []
        if self.decision_lines is None:
            self.decision_lines = []
        if self.telemetry_lines is None:
            self.telemetry_lines = []
        if self.trajectory_points is None:
            self.trajectory_points = []
        if self.waypoints is None:
            self.waypoints = []


class _CardFrame(QFrame):
    def __init__(self, title: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("dashboardCard")
        self.setStyleSheet(
            "QFrame#dashboardCard { background: #0d1726; border: 1px solid #22364e; border-radius: 16px; }"
        )
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(12, 10, 12, 12)
        self.layout.setSpacing(8)
        if title is not None:
            self.title_label = QLabel(title, self)
            self.title_label.setObjectName("panelTitle")
            self.layout.addWidget(self.title_label)


class _VideoPanel(_CardFrame):
    def __init__(self, title: str, placeholder_title: str, placeholder_subtitle: str = "", size: tuple[int, int] | None = None, parent=None) -> None:
        super().__init__(title, parent)
        self.placeholder_title = placeholder_title
        self.placeholder_subtitle = placeholder_subtitle
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background: #050a11; border: 1px solid #2b4565; border-radius: 12px;")
        if size is not None:
            self.video_label.setMinimumSize(*size)
        else:
            self.video_label.setMinimumSize(320, 220)
        self.layout.addWidget(self.video_label)

    def set_frame(self, frame_bgr, fallback_size: tuple[int, int] | None = None) -> None:
        if frame_bgr is None:
            if fallback_size is None:
                fallback_size = (640, 360)
            frame_bgr = _build_placeholder_frame(fallback_size[0], fallback_size[1], self.placeholder_title, self.placeholder_subtitle)
        pixmap = _to_qpixmap_bgr(frame_bgr)
        if pixmap is None:
            self.video_label.setText(self.placeholder_title)
            return
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)


class _MetricCard(QFrame):
    def __init__(self, title: str, value: str = "--", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("dashboardCard")
        self.setStyleSheet(
            "QFrame#dashboardCard { background: #0d1726; border: 1px solid #22364e; border-radius: 14px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        self.title_label = QLabel(title, self)
        self.title_label.setObjectName("metricName")
        self.value_label = QLabel(value, self)
        self.value_label.setObjectName("metricValue")
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class _LivePlotCard(_CardFrame):
    def __init__(self, title: str, series_names: list[str], colors: list[str], y_label: str | None = None, parent=None) -> None:
        if pg is None:
            raise RuntimeError("pyqtgraph is required for the tabbed analytics views. Install pyqtgraph to continue.")
        super().__init__(title, parent)
        self.plot = pg.PlotWidget(background="#08111d")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.setAntialiasing(True)
        self.plot.getAxis("left").setTextPen(pg.mkPen("#8fb3d9"))
        self.plot.getAxis("bottom").setTextPen(pg.mkPen("#8fb3d9"))
        if y_label:
            self.plot.setLabel("left", y_label, color="#b7cde6")
        self.plot.setLabel("bottom", "Samples", color="#b7cde6")
        self.curves: dict[str, object] = {}
        for series_name, color in zip(series_names, colors, strict=False):
            self.curves[series_name] = self.plot.plot([], [], pen=pg.mkPen(color, width=2), name=series_name)
        self.layout.addWidget(self.plot)

    def set_series(self, x_values, series_values: dict[str, list[float]]) -> None:
        for series_name, values in series_values.items():
            curve = self.curves.get(series_name)
            if curve is not None:
                curve.setData(x_values, values)


class DashboardTab(QWidget):
    def __init__(self, on_traffic_toggled, parent=None) -> None:
        super().__init__(parent)
        self.on_traffic_toggled = on_traffic_toggled

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        self.carla_panel = _VideoPanel("CARLA ROAD VIEW", "CARLA VIEW", "Waiting for simulator frame", size=(960, 540), parent=self)
        root.addWidget(self.carla_panel, stretch=3)
        self._minimap_size = 220
        self._minimap_circular = True

        self.minimap_container = QFrame(self.carla_panel.video_label)
        self.minimap_container.setObjectName("miniMapHud")
        self.minimap_container.setStyleSheet(
            "QFrame#miniMapHud { background: rgba(8, 12, 20, 220); border: 2px solid #4b6c90; border-radius: 110px; }"
        )
        self.minimap_label = QLabel(self.minimap_container)
        self.minimap_label.setAlignment(Qt.AlignCenter)
        self.minimap_label.setStyleSheet("background: transparent;")
        self._position_minimap()
        self.minimap_container.show()

        sidebar = _CardFrame("DRIVING SUMMARY", self)
        sidebar.layout.setSpacing(10)
        root.addWidget(sidebar, stretch=1)

        self.webcam_panel = _VideoPanel("WEBCAM MONITOR", "WEBCAM", "Driver monitoring feed", size=(320, 260), parent=self)
        sidebar.layout.addWidget(self.webcam_panel)

        status_grid = QGridLayout()
        status_grid.setHorizontalSpacing(8)
        status_grid.setVerticalSpacing(8)
        sidebar.layout.addLayout(status_grid)

        self.driver_state_chip = QLabel("ALERT", self)
        self.mode_chip = QLabel("MANUAL", self)
        self.speed_chip = QLabel("0.0 km/h", self)
        self.fps_chip = QLabel("0.0 / 0.0 FPS", self)
        self.camera_chip = QLabel("Camera", self)
        self.lane_chip = QLabel("Lane hint: N/A", self)

        self._status_widgets = [
            self.driver_state_chip,
            self.mode_chip,
            self.speed_chip,
            self.fps_chip,
            self.camera_chip,
            self.lane_chip,
        ]

        status_grid.addWidget(self.driver_state_chip, 0, 0)
        status_grid.addWidget(self.mode_chip, 0, 1)
        status_grid.addWidget(self.speed_chip, 1, 0)
        status_grid.addWidget(self.fps_chip, 1, 1)
        status_grid.addWidget(self.camera_chip, 2, 0)
        status_grid.addWidget(self.lane_chip, 2, 1)

        self.traffic_button = QPushButton("Traffic: OFF", self)
        self.traffic_button.setCheckable(True)
        self.traffic_button.toggled.connect(on_traffic_toggled)
        self.traffic_button.setStyleSheet(
            """
            QPushButton {
                background: #5d1e2a;
                border: 1px solid #ab4257;
                border-radius: 10px;
                padding: 8px 12px;
                color: #ffeef2;
                font-weight: 700;
            }
            QPushButton:checked {
                background: #1a6a42;
                border: 1px solid #2ea769;
                color: #eafff3;
            }
            """
        )
        sidebar.layout.addWidget(self.traffic_button)

        self.minimap_mode_button = QPushButton("MiniMap: Circle", self)
        self.minimap_mode_button.setCheckable(True)
        self.minimap_mode_button.setChecked(True)
        self.minimap_mode_button.setStyleSheet(
            "QPushButton { background: #1a3a52; border: 1px solid #3a6a92; border-radius: 10px; padding: 8px 12px; color: #eef7ff; font-weight: 700; }"
            "QPushButton:checked { background: #16425f; border: 1px solid #5b8db8; }"
        )
        self.minimap_mode_button.toggled.connect(self._toggle_minimap_shape)
        sidebar.layout.addWidget(self.minimap_mode_button)

        self.controls_label = QLabel(
            "Controls: W/A/S/D, arrows, SPACE brake, P autopilot, TAB camera, BACKSPACE switch vehicle, F1 HUD, H help",
            self,
        )
        self.controls_label.setWordWrap(True)
        self.controls_label.setStyleSheet(
            "background: #0a1220; border: 1px solid #2b4565; border-radius: 10px; padding: 10px; color: #9dc1e6;"
        )
        sidebar.layout.addWidget(self.controls_label)

        sidebar.layout.addStretch(1)

    def set_frames(self, carla_frame, webcam_frame) -> None:
        self.carla_panel.set_frame(carla_frame, fallback_size=(960, 540))
        self.webcam_panel.set_frame(webcam_frame, fallback_size=(320, 260))
        self._position_minimap()

    def _toggle_minimap_shape(self, checked: bool) -> None:
        self._minimap_circular = checked
        self.minimap_mode_button.setText("MiniMap: Circle" if checked else "MiniMap: Rect")
        radius = self._minimap_size // 2 if checked else 16
        self.minimap_container.setStyleSheet(
            f"QFrame#miniMapHud {{ background: rgba(8, 12, 20, 220); border: 2px solid #4b6c90; border-radius: {radius}px; }}"
        )
        if checked and QRegion is not None and QRect is not None:
            self.minimap_container.setMask(QRegion(QRect(0, 0, self._minimap_size, self._minimap_size), QRegion.Ellipse))
        else:
            self.minimap_container.clearMask()

    def _position_minimap(self) -> None:
        margin = 16
        size = self._minimap_size
        x_pos = margin
        y_pos = max(margin, self.carla_panel.video_label.height() - size - margin)
        self.minimap_container.setGeometry(x_pos, y_pos, size, size)
        self.minimap_label.setGeometry(0, 0, size, size)
        if self._minimap_circular and QRegion is not None and QRect is not None:
            self.minimap_container.setMask(QRegion(QRect(0, 0, size, size), QRegion.Ellipse))
        else:
            self.minimap_container.clearMask()

    def set_minimap(
        self,
        map_waypoints: list[dict],
        vehicle_location,
        route_points: list[tuple[float, float, float]] | None,
        destination: tuple[float, float, float] | None,
        heading_deg: float,
    ) -> None:
        minimap_frame = _build_minimap_frame(
            map_waypoints=map_waypoints,
            vehicle_location=vehicle_location,
            route_points=route_points,
            destination=destination,
            heading_deg=heading_deg,
            size=self._minimap_size,
            meters_radius=60.0,
        )
        pixmap = _to_qpixmap_bgr(minimap_frame)
        if pixmap is None:
            return
        scaled = pixmap.scaled(
            self.minimap_label.size(),
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation,
        )
        self.minimap_label.setPixmap(scaled)

    def set_snapshot(self, snapshot: DashboardSnapshot) -> None:
        state_kind = "good" if snapshot.driver_state == "ALERT" else "warn" if snapshot.driver_state == "DROWSY" else "bad"
        _set_badge(self.driver_state_chip, f"{snapshot.driver_state}", state_kind)

        mode_kind = "info" if snapshot.mode == "AUTOPILOT" else "warn" if snapshot.mode == "PARKING" else "neutral"
        _set_badge(self.mode_chip, snapshot.mode, mode_kind)

        _set_badge(self.speed_chip, f"{snapshot.speed_kmh:.1f} km/h", "info")
        _set_badge(self.fps_chip, f"{snapshot.client_fps:.1f} / {snapshot.server_fps:.1f} FPS", "neutral")
        _set_badge(self.camera_chip, snapshot.camera_label, "neutral")
        _set_badge(self.lane_chip, snapshot.lane_hint, "neutral")

    def set_traffic_running(self, running: bool) -> None:
        self.traffic_button.blockSignals(True)
        self.traffic_button.setChecked(running)
        self.traffic_button.blockSignals(False)
        self.traffic_button.setText("Traffic: ON" if running else "Traffic: OFF")


class DriverMonitoringTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        if pg is None:
            raise RuntimeError("pyqtgraph is required for the driver monitoring graph.")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        root.addLayout(top_row)

        self.webcam_panel = _VideoPanel("WEBCAM FEED", "WEBCAM", "Driver monitoring input", size=(420, 320), parent=self)
        top_row.addWidget(self.webcam_panel, stretch=1)

        metrics_card = _CardFrame("DROWSINESS METRICS", self)
        metrics_card.layout.setSpacing(10)
        top_row.addWidget(metrics_card, stretch=1)

        metric_grid = QGridLayout()
        metric_grid.setHorizontalSpacing(10)
        metric_grid.setVerticalSpacing(10)
        metrics_card.layout.addLayout(metric_grid)

        self.ear_metric = _MetricCard("EAR", "0.000", self)
        self.score_metric = _MetricCard("Drowsiness Score", "0", self)
        self.state_metric = _MetricCard("State", "ALERT", self)
        self.status_metric = _MetricCard("Status", "NO FACE", self)
        metric_grid.addWidget(self.ear_metric, 0, 0)
        metric_grid.addWidget(self.score_metric, 0, 1)
        metric_grid.addWidget(self.state_metric, 1, 0)
        metric_grid.addWidget(self.status_metric, 1, 1)

        self.ear_plot = _LivePlotCard("EAR OVER TIME", ["EAR"], ["#2db5ff"], y_label="EAR", parent=self)
        root.addWidget(self.ear_plot)

        logs_card = _CardFrame("ALERT LOGS", self)
        self.alert_log = QPlainTextEdit(self)
        self.alert_log.setReadOnly(True)
        self.alert_log.setMaximumBlockCount(200)
        self.alert_log.setStyleSheet("background: #0a111c; border: 1px solid #213249; border-radius: 12px; padding: 8px; color: #d4e4f8;")
        logs_card.layout.addWidget(self.alert_log)
        root.addWidget(logs_card, stretch=1)

    def set_frame(self, webcam_frame) -> None:
        self.webcam_panel.set_frame(webcam_frame, fallback_size=(420, 320))

    def set_snapshot(self, snapshot: DashboardSnapshot, ear_history: deque[float]) -> None:
        self.ear_metric.set_value(f"{snapshot.ear:.3f}")
        self.score_metric.set_value(str(snapshot.drowsiness_score))
        self.state_metric.set_value(snapshot.driver_state)
        self.status_metric.set_value(snapshot.telemetry_lines[0] if snapshot.telemetry_lines else "NO FACE")

        x_values = list(range(len(ear_history)))
        self.ear_plot.set_series(x_values, {"EAR": list(ear_history)})

        if snapshot.alert_lines:
            self.alert_log.setPlainText("\n".join(snapshot.alert_lines[-120:]))


class VehicleAnalyticsTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        if pg is None:
            raise RuntimeError("pyqtgraph is required for the vehicle analytics graphs.")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        metrics_card = _CardFrame("VEHICLE STATE", self)
        metric_grid = QGridLayout()
        metric_grid.setHorizontalSpacing(10)
        metric_grid.setVerticalSpacing(10)
        metrics_card.layout.addLayout(metric_grid)

        self.speed_metric = _MetricCard("Speed", "0.0 km/h", self)
        self.steer_metric = _MetricCard("Steer", "+0.00", self)
        self.throttle_metric = _MetricCard("Throttle", "0.00", self)
        self.brake_metric = _MetricCard("Brake", "0.00", self)
        self.gear_metric = _MetricCard("Gear", "1", self)
        self.reverse_metric = _MetricCard("Reverse", "OFF", self)
        metric_grid.addWidget(self.speed_metric, 0, 0)
        metric_grid.addWidget(self.steer_metric, 0, 1)
        metric_grid.addWidget(self.throttle_metric, 1, 0)
        metric_grid.addWidget(self.brake_metric, 1, 1)
        metric_grid.addWidget(self.gear_metric, 2, 0)
        metric_grid.addWidget(self.reverse_metric, 2, 1)
        root.addWidget(metrics_card)

        plot_row = QHBoxLayout()
        plot_row.setSpacing(12)
        root.addLayout(plot_row, stretch=1)

        self.speed_plot = _LivePlotCard("SPEED vs TIME", ["Speed"], ["#2db5ff"], y_label="km/h", parent=self)
        self.steer_plot = _LivePlotCard("STEERING vs TIME", ["Steer"], ["#8fd14f"], y_label="steer", parent=self)
        self.pedal_plot = _LivePlotCard("THROTTLE / BRAKE", ["Throttle", "Brake"], ["#ffb347", "#ff6b6b"], y_label="value", parent=self)
        plot_row.addWidget(self.speed_plot, stretch=1)
        plot_row.addWidget(self.steer_plot, stretch=1)
        plot_row.addWidget(self.pedal_plot, stretch=1)

    def set_snapshot(self, snapshot: DashboardSnapshot, speed_history: deque[float], steer_history: deque[float], throttle_history: deque[float], brake_history: deque[float]) -> None:
        self.speed_metric.set_value(f"{snapshot.speed_kmh:.1f} km/h")
        self.steer_metric.set_value(f"{snapshot.steer:+.2f}")
        self.throttle_metric.set_value(f"{snapshot.throttle:.2f}")
        self.brake_metric.set_value(f"{snapshot.brake:.2f}")
        self.gear_metric.set_value(str(snapshot.gear))
        self.reverse_metric.set_value("ON" if snapshot.reverse_enabled else "OFF")

        x_values = list(range(len(speed_history)))
        self.speed_plot.set_series(x_values, {"Speed": list(speed_history)})
        self.steer_plot.set_series(x_values, {"Steer": list(steer_history)})
        self.pedal_plot.set_series(x_values, {"Throttle": list(throttle_history), "Brake": list(brake_history)})


class AutonomousParkingTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        if pg is None:
            raise RuntimeError("pyqtgraph is required for the parking trajectory graph.")

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        left_column = QVBoxLayout()
        left_column.setSpacing(12)
        root.addLayout(left_column, stretch=3)

        self.preview_panel = _VideoPanel("AUTONOMOUS / PARKING VIEW", "TOP VIEW", "Placeholder or CARLA frame", size=(720, 360), parent=self)
        left_column.addWidget(self.preview_panel)

        self.state_card = _CardFrame("CURRENT STATE", self)
        self.state_card.layout.setSpacing(8)
        left_column.addWidget(self.state_card)

        state_row = QHBoxLayout()
        state_row.setSpacing(8)
        self.state_badges: dict[str, object] = {}
        for state_name in ["Manual", "Autonomous Driving", "Searching Parking", "Parking Complete"]:
            badge = QLabel(state_name, self)
            badge.setAlignment(Qt.AlignCenter)
            badge.setStyleSheet(_badge_style("neutral"))
            self.state_badges[state_name] = badge
            state_row.addWidget(badge)
        self.state_card.layout.addLayout(state_row)

        self.active_state_label = QLabel("Manual", self)
        self.active_state_label.setObjectName("metricValue")
        self.state_card.layout.addWidget(self.active_state_label)

        self.lane_card = _CardFrame("LANE STATUS", self)
        lane_grid = QGridLayout()
        lane_grid.setHorizontalSpacing(10)
        lane_grid.setVerticalSpacing(10)
        self.lane_card.layout.addLayout(lane_grid)
        self.current_lane_metric = _MetricCard("Current Lane", "N/A", self)
        self.left_lane_metric = _MetricCard("Left Lane", "N/A", self)
        self.right_lane_metric = _MetricCard("Right Lane", "N/A", self)
        lane_grid.addWidget(self.current_lane_metric, 0, 0)
        lane_grid.addWidget(self.left_lane_metric, 0, 1)
        lane_grid.addWidget(self.right_lane_metric, 1, 0, 1, 2)
        left_column.addWidget(self.lane_card)

        self.trajectory_plot = _LivePlotCard("TRAJECTORY", ["Path"], ["#2db5ff"], y_label="y", parent=self)
        left_column.addWidget(self.trajectory_plot, stretch=1)

        right_column = _CardFrame("DECISION LOGS", self)
        right_column.layout.setSpacing(10)
        root.addWidget(right_column, stretch=2)

        self.decision_log = QPlainTextEdit(self)
        self.decision_log.setReadOnly(True)
        self.decision_log.setMaximumBlockCount(200)
        self.decision_log.setStyleSheet("background: #0a111c; border: 1px solid #213249; border-radius: 12px; padding: 8px; color: #d4e4f8;")
        right_column.layout.addWidget(self.decision_log, stretch=1)

    def set_frame(self, frame_bgr) -> None:
        self.preview_panel.set_frame(frame_bgr, fallback_size=(720, 360))

    def set_snapshot(self, snapshot: DashboardSnapshot) -> None:
        active_state = snapshot.parking_state or "Manual"
        self.active_state_label.setText(active_state)
        for state_name, badge in self.state_badges.items():
            if state_name == active_state:
                kind = "good" if state_name == "Manual" else "info" if state_name == "Autonomous Driving" else "warn" if state_name == "Searching Parking" else "bad"
                badge.setStyleSheet(_badge_style(kind))
            else:
                badge.setStyleSheet(_badge_style("neutral"))

        current_lane = "N/A"
        left_lane = "N/A"
        right_lane = "N/A"
        for line in snapshot.telemetry_lines:
            if line.startswith("Lane:"):
                current_lane = line.replace("Lane:", "", 1).strip()
            elif line.startswith("Neighbors:"):
                parts = line.replace("Neighbors:", "", 1).strip().split()
                for part in parts:
                    if part.startswith("L="):
                        left_lane = part[2:]
                    elif part.startswith("R="):
                        right_lane = part[2:]

        self.current_lane_metric.set_value(current_lane)
        self.left_lane_metric.set_value(left_lane)
        self.right_lane_metric.set_value(right_lane)

        points = snapshot.trajectory_points
        if points:
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            self.trajectory_plot.set_series(x_values, {"Path": y_values})

        if snapshot.decision_lines:
            self.decision_log.setPlainText("\n".join(snapshot.decision_lines[-120:]))


class WaypointsTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        title_card = _CardFrame("NAVIGATION WAYPOINTS", self)
        title_layout = QVBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_card.layout.addLayout(title_layout)

        info_label = QLabel(
            "CARLA camera view with waypoint path overlay. Green line shows the planned route ahead, "
            "with distance markers for each waypoint."
        )
        info_label.setStyleSheet("color: #a0aec0; font-size: 12px;")
        info_label.setWordWrap(True)
        title_layout.addWidget(info_label)
        root.addWidget(title_card)

        # Video display card
        video_card = _CardFrame("NAVIGATION CAMERA VIEW", self)
        video_layout = QVBoxLayout()
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_card.layout.addLayout(video_layout)

        self.video_label = QLabel()
        self.video_label.setMinimumHeight(480)
        self.video_label.setStyleSheet("background: #0a0e1a; border: 1px solid #22364e;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Waiting for CARLA camera feed...")
        video_layout.addWidget(self.video_label)
        root.addWidget(video_card, stretch=1)

    def set_frame(self, frame: np.ndarray | None) -> None:
        """Display CARLA frame with waypoint overlay."""
        if frame is None:
            self.video_label.setText("No camera feed available")
            self.video_label.setPixmap(QPixmap())
            return

        try:
            # Ensure frame is C-contiguous and correct dtype
            if frame.dtype != np.uint8:
                frame = np.uint8(np.clip(frame, 0, 255))
            frame = np.ascontiguousarray(frame)

            # Convert BGR to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame

            # Convert to QImage using tobytes() for proper memory handling
            h, w = frame_rgb.shape[:2]
            bytes_per_line = 3 * w
            qt_image = QImage(frame_rgb.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)

            # Convert to pixmap and display
            pixmap = QPixmap.fromImage(qt_image)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaledToWidth(960, Qt.SmoothTransformation)
                self.video_label.setPixmap(scaled_pixmap)
                self.video_label.setText("")  # Clear any placeholder text
            else:
                self.video_label.setText("Error converting frame to pixmap")
                self.video_label.setPixmap(QPixmap())
        except Exception:
            self.video_label.setText("Error processing camera frame")
            self.video_label.setPixmap(QPixmap())

    def set_snapshot(self, snapshot: DashboardSnapshot) -> None:
        """Update with snapshot data (for compatibility)."""
        pass


class MapVisualizationTab(QWidget):
    def __init__(
        self,
        on_destination_selected,
        on_plan_route,
        on_start_route,
        on_stop_route,
        on_clear_destination,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if pg is None:
            raise RuntimeError("pyqtgraph is required for the map visualization tab.")

        self.on_destination_selected = on_destination_selected
        self.on_plan_route = on_plan_route
        self.on_start_route = on_start_route
        self.on_stop_route = on_stop_route
        self.on_clear_destination = on_clear_destination
        self._has_autoranged = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        header_card = _CardFrame("LOADED MAP", self)
        header_card.layout.setSpacing(8)
        info_label = QLabel(
            "Click on the map to set a destination. Use Plan/Start controls to route the ego vehicle to the selected point."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #a0aec0; font-size: 12px;")
        header_card.layout.addWidget(info_label)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(8)
        self.plan_button = QPushButton("Plan Route", self)
        self.start_button = QPushButton("Start Route", self)
        self.stop_button = QPushButton("Stop Route", self)
        self.clear_button = QPushButton("Clear Destination", self)
        self.plan_button.clicked.connect(on_plan_route)
        self.start_button.clicked.connect(on_start_route)
        self.stop_button.clicked.connect(on_stop_route)
        self.clear_button.clicked.connect(on_clear_destination)
        controls_row.addWidget(self.plan_button)
        controls_row.addWidget(self.start_button)
        controls_row.addWidget(self.stop_button)
        controls_row.addWidget(self.clear_button)
        controls_row.addStretch(1)
        header_card.layout.addLayout(controls_row)

        self.summary_label = QLabel("Waypoints: 0 | Resolution: 2.0 m | Route: idle", self)
        self.summary_label.setObjectName("metricValue")
        header_card.layout.addWidget(self.summary_label)
        root.addWidget(header_card)

        plot_card = _CardFrame("WAYPOINT CLOUD", self)
        plot_card.layout.setSpacing(8)
        self.plot = pg.PlotWidget(background="#08111d")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.setAntialiasing(True)
        self.plot.setAspectLocked(True, ratio=1)
        self.plot.setLabel("bottom", "X (m)", color="#b7cde6")
        self.plot.setLabel("left", "Y (m)", color="#b7cde6")
        self.plot.getAxis("left").setTextPen(pg.mkPen("#8fb3d9"))
        self.plot.getAxis("bottom").setTextPen(pg.mkPen("#8fb3d9"))
        self.plot.scene().sigMouseClicked.connect(self._on_plot_clicked)

        self.waypoint_scatter = pg.ScatterPlotItem(size=4, pen=pg.mkPen(None), brush=pg.mkBrush("#2db5ff"))
        self.junction_scatter = pg.ScatterPlotItem(size=6, pen=pg.mkPen("#ffd166", width=1), brush=pg.mkBrush("#f4d03f"))
        self.rightmost_scatter = pg.ScatterPlotItem(size=6, pen=pg.mkPen("#ffb347", width=1), brush=pg.mkBrush("#ff6b35"))
        self.right_offset_scatter = pg.ScatterPlotItem(size=5, pen=pg.mkPen("#9a77ff", width=1), brush=pg.mkBrush("#c9b6ff"))
        self.vehicle_scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen("#ffcf4d", width=2), brush=pg.mkBrush("#ffb347"))
        self.destination_scatter = pg.ScatterPlotItem(size=14, pen=pg.mkPen("#ff4d6d", width=2), brush=pg.mkBrush("#ff8096"))
        self.route_curve = self.plot.plot([], [], pen=pg.mkPen("#8fd14f", width=2))
        self.plot.addItem(self.waypoint_scatter)
        self.plot.addItem(self.junction_scatter)
        self.plot.addItem(self.rightmost_scatter)
        self.plot.addItem(self.right_offset_scatter)
        self.plot.addItem(self.route_curve)
        self.plot.addItem(self.vehicle_scatter)
        self.plot.addItem(self.destination_scatter)

        plot_card.layout.addWidget(self.plot)
        root.addWidget(plot_card, stretch=1)

    def _on_plot_clicked(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        view_pos = self.plot.plotItem.vb.mapSceneToView(event.scenePos())
        x_world = float(view_pos.x())
        y_world = -float(view_pos.y())
        self.on_destination_selected(x_world, y_world)

    def set_map_data(
        self,
        waypoints: list[dict],
        vehicle_location=None,
        map_name: str = "Loaded map",
        resolution: float = 2.0,
        destination: tuple[float, float, float] | None = None,
        route_points: list[tuple[float, float, float]] | None = None,
        route_status: str = "idle",
    ) -> None:
        waypoint_x = [
            float(wp["location"][0])
            for wp in waypoints
            if wp.get("location") is not None
            and not bool(wp.get("is_junction", False))
            and not bool(wp.get("is_rightmost_lane", False))
            and not bool(wp.get("is_right_offset", False))
        ]
        waypoint_y = [
            -float(wp["location"][1])
            for wp in waypoints
            if wp.get("location") is not None
            and not bool(wp.get("is_junction", False))
            and not bool(wp.get("is_rightmost_lane", False))
            and not bool(wp.get("is_right_offset", False))
        ]
        self.waypoint_scatter.setData(waypoint_x, waypoint_y)

        junction_x = [
            float(wp["location"][0])
            for wp in waypoints
            if wp.get("location") is not None and bool(wp.get("is_junction", False))
        ]
        junction_y = [
            -float(wp["location"][1])
            for wp in waypoints
            if wp.get("location") is not None and bool(wp.get("is_junction", False))
        ]
        self.junction_scatter.setData(junction_x, junction_y)

        rightmost_x = [
            float(wp["location"][0])
            for wp in waypoints
            if wp.get("location") is not None
            and not bool(wp.get("is_junction", False))
            and bool(wp.get("is_rightmost_lane", False))
        ]
        rightmost_y = [
            -float(wp["location"][1])
            for wp in waypoints
            if wp.get("location") is not None
            and not bool(wp.get("is_junction", False))
            and bool(wp.get("is_rightmost_lane", False))
        ]
        self.rightmost_scatter.setData(rightmost_x, rightmost_y)

        right_offset_x = [
            float(wp["location"][0])
            for wp in waypoints
            if wp.get("location") is not None
            and not bool(wp.get("is_junction", False))
            and bool(wp.get("is_right_offset", False))
        ]
        right_offset_y = [
            -float(wp["location"][1])
            for wp in waypoints
            if wp.get("location") is not None
            and not bool(wp.get("is_junction", False))
            and bool(wp.get("is_right_offset", False))
        ]
        self.right_offset_scatter.setData(right_offset_x, right_offset_y)

        if (waypoint_x or rightmost_x or junction_x or right_offset_x) and not self._has_autoranged:
            self.plot.autoRange()
            self._has_autoranged = True

        if vehicle_location is not None:
            self.vehicle_scatter.setData([float(vehicle_location.x)], [-float(vehicle_location.y)])
        else:
            self.vehicle_scatter.setData([], [])

        if destination is not None:
            self.destination_scatter.setData([float(destination[0])], [-float(destination[1])])
        else:
            self.destination_scatter.setData([], [])

        route_points = route_points or []
        if route_points:
            route_x = [float(point[0]) for point in route_points]
            route_y = [-float(point[1]) for point in route_points]
            self.route_curve.setData(route_x, route_y)
        else:
            self.route_curve.setData([], [])

        self.summary_label.setText(
            f"{map_name} | Waypoints: {len(waypoints)} | Junctions: {len(junction_x)} | Rightmost: {len(rightmost_x)} | Right+330cm: {len(right_offset_x)} | Resolution: {resolution:.1f} m | Route: {route_status}"
        )



class TabbedDashboardWindow(QMainWindow):
    def __init__(
        self,
        controller: CarlaSafetyController,
        camera_index: int,
        carla_view_width: int,
        carla_view_height: int,
        carla_camera_mode: str,
        carla_view_index: int,
        use_mediapipe: bool = False,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Driver Monitoring Dashboard (Tabbed PyQt)")
        self.resize(1760, 1040)
        self.setFocusPolicy(Qt.StrongFocus)

        self.controller = controller
        self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

        self.detector = DrowsinessDetector()
        self.alerts = AlertController()
        self._use_mediapipe = use_mediapipe
        self._mediapipe_system = None

        if self._use_mediapipe:
            try:
                from .mediapipe_system import MediaPipeDrowsinessSystem
            except ImportError as exc:
                raise RuntimeError(
                    "MediaPipe backend requested but unavailable. Install dependencies and retry."
                ) from exc
            self._mediapipe_system = MediaPipeDrowsinessSystem()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

        self.autopilot_enabled = False
        self.reverse_enabled = False
        self.manual_gear_shift = False
        self.current_gear = 1
        self.steer_cache = 0.0
        self.show_hud = True
        self.show_help = False
        self._fullscreen = False
        self._pressed_keys: set[int] = set()
        self._last_tick_time: float | None = None
        self._client_fps_smoothed: float = 0.0
        self._traffic_process: subprocess.Popen | None = None
        self._lights_state = int(carla.VehicleLightState.NONE) if carla is not None else 0
        self._latest_webcam_frame = None
        self._latest_carla_frame = None
        self._latest_trajectory: deque[tuple[float, float]] = deque(maxlen=200)
        self._cached_waypoints: list[dict] = []
        self._parking_state = "Manual"

        self.ear_history: deque[float] = deque(maxlen=180)
        self.speed_history: deque[float] = deque(maxlen=180)
        self.steer_history: deque[float] = deque(maxlen=180)
        self.throttle_history: deque[float] = deque(maxlen=180)
        self.brake_history: deque[float] = deque(maxlen=180)
        self.alert_log: deque[str] = deque(maxlen=120)
        self.decision_log: deque[str] = deque(maxlen=120)
        self._sleep_safety_engaged = False
        self._sleep_safety_attempted = False
        self._sleep_detected_since: float | None = None
        self._sleep_safety_delay_seconds = 3.0
        self._sleep_safety_pending_logged = False
        self._sleep_decision_overlay = None
        self._sleep_decision_countdown_label = None
        self._sleep_decision_expires_at: float | None = None
        self._sleep_decision_visible = False
        self._sleep_right_shift_pending = False
        self._sleep_right_shift_started_at: float | None = None
        self._sleep_right_shift_last_request_at: float = 0.0
        self._sleep_right_shift_timeout_s: float = 8.0
        self._sleep_right_shift_rightmost_since: float | None = None
        self._sleep_right_shift_hold_s: float = 3.0
        self._status_message: str = ""
        self._status_severity: str = "neutral"
        self._status_updated_at: float = 0.0
        self._status_source: str = ""
        self._drowsiness_log_last_at: float = 0.0
        self._drowsiness_log_last_state: DrowsinessState | None = None
        self._last_route_status = self.controller.get_route_status()
        self._startup_sound_checked = False
        self._arrival_popup = None

        self.camera_mode = self.controller.set_camera_mode(carla_camera_mode) if self.controller.available() else carla_camera_mode
        self.camera_view_index = carla_view_index if self.camera_mode == "rgb" else 0
        if self.controller.available() and self.controller.world is not None and self.controller.vehicle is not None:
            self.controller.start_rgb_preview(
                width=carla_view_width,
                height=carla_view_height,
                transform_index=self.camera_view_index,
                sensor_mode=self.camera_mode,
            )

        self.setStyleSheet(
            """
            QWidget {
                background: #070d16;
                color: #dbe7f7;
                font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
                font-size: 13px;
            }
            QPushButton {
                background: #163252;
                border: 1px solid #2d80c6;
                border-radius: 10px;
                padding: 8px 12px;
                color: #eef7ff;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #173659;
            }
            QPushButton:pressed {
                background: #0d1726;
            }
            QPushButton:disabled {
                background: #1a2230;
                border: 1px solid #33445a;
                color: #93aecb;
            }
            QLabel#title {
                font-size: 24px;
                font-weight: 700;
                color: #f1f6ff;
            }
            QTabWidget::pane {
                border: 0;
                margin-top: 8px;
            }
            QTabBar::tab {
                background: #0d1726;
                border: 1px solid #22364e;
                border-bottom: none;
                padding: 8px 14px;
                margin-right: 3px;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                color: #a8bfd9;
                font-weight: 700;
                font-size: 12px;
                min-width: 0;
            }
            QTabBar::tab:hover {
                background: #1a2230;
                color: #dbe7f7;
            }
            QTabBar::tab:selected {
                background: #173659;
                color: #f1f6ff;
            }
            QLabel#panelTitle {
                font-size: 13px;
                color: #93aecb;
                font-weight: 700;
                letter-spacing: 0.4px;
            }
            QLabel#metricName {
                color: #89a2bc;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#metricValue {
                color: #f7fbff;
                font-size: 18px;
                font-weight: 700;
            }
            """
        )

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title_card = _CardFrame(None, self)
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(10)
        title_card.layout.addLayout(title_row)

        title_block = QVBoxLayout()
        title_block.setSpacing(4)
        title_row.addLayout(title_block, stretch=1)
        self.header_label = QLabel("Driver Monitoring & Autonomous Safety Dashboard", self)
        self.header_label.setObjectName("title")
        title_block.addWidget(self.header_label)
        self.subheader_label = QLabel("CARLA-integrated tabbed PyQt5 dashboard with live analytics and monitoring", self)
        self.subheader_label.setStyleSheet("color: #8aa5c4;")
        title_block.addWidget(self.subheader_label)

        self.global_status = QLabel("READY", self)
        self.global_status.setAlignment(Qt.AlignCenter)
        title_row.addWidget(self.global_status)
        _set_badge(self.global_status, "READY", "good")

        backend_button = QPushButton("Detection: HAAR", self)
        backend_button.setMaximumWidth(160)
        backend_button.setStyleSheet(
            "QPushButton { background: #1a3a52; border: 1px solid #3a6a92; border-radius: 6px; padding: 6px 12px; color: #fff; font-weight: 600; }"
            "QPushButton:hover { background: #2a5a72; }"
            "QPushButton:pressed { background: #1a2a42; }"
        )
        backend_button.clicked.connect(self._toggle_detection_backend)
        self.backend_button = backend_button
        title_row.addWidget(backend_button)
        root.addWidget(title_card)

        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(True)
        self.tabs.tabBar().setElideMode(Qt.ElideRight)
        root.addWidget(self.tabs, stretch=1)

        self.dashboard_tab = DashboardTab(self._on_traffic_toggled, self)
        self.monitoring_tab = DriverMonitoringTab(self)
        self.analytics_tab = VehicleAnalyticsTab(self)
        self.parking_tab = AutonomousParkingTab(self)
        self.map_tab = MapVisualizationTab(
            on_destination_selected=self._on_map_destination_selected,
            on_plan_route=self._on_map_plan_route,
            on_start_route=self._on_map_start_route,
            on_stop_route=self._on_map_stop_route,
            on_clear_destination=self._on_map_clear_destination,
            parent=self,
        )

        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        self.tabs.setTabToolTip(0, "Dashboard")
        self.tabs.addTab(self.monitoring_tab, "Monitoring")
        self.tabs.setTabToolTip(1, "Driver Monitoring")
        self.tabs.addTab(self.analytics_tab, "Analytics")
        self.tabs.setTabToolTip(2, "Vehicle Analytics")
        self.tabs.addTab(self.parking_tab, "Parking")
        self.tabs.setTabToolTip(3, "Autonomous / Parking")
        self.tabs.addTab(self.map_tab, "Map")
        self.tabs.setTabToolTip(4, "Loaded Map Waypoints")

        self._create_sleep_decision_overlay(central)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)
        QTimer.singleShot(800, self._run_startup_sound_check)

        self.activateWindow()
        self.setFocus()

    def resizeEvent(self, event):
        try:
            super().resizeEvent(event)
        except Exception:
            pass
        self._position_sleep_decision_overlay()

    def _create_sleep_decision_overlay(self, parent) -> None:
        if QFrame is None or QVBoxLayout is None or QLabel is None or QPushButton is None:
            return

        overlay = QFrame(parent)
        overlay.setObjectName("sleepDecisionOverlay")
        overlay.setVisible(False)
        overlay.setStyleSheet(
            "QFrame#sleepDecisionOverlay { background: rgba(7, 13, 22, 235); border: 2px solid #a43a57; border-radius: 14px; }"
            "QLabel { color: #f1f6ff; }"
            "QLabel#sleepTitle { font-size: 18px; font-weight: 800; }"
            "QLabel#sleepSubtitle { font-size: 13px; color: #dbe7f7; }"
            "QLabel#sleepCountdown { font-size: 13px; color: #ffd6e0; font-weight: 700; }"
            "QPushButton { background: #173659; border: 1px solid #2d80c6; border-radius: 10px; padding: 10px 14px; color: #f1f6ff; font-weight: 800; }"
            "QPushButton:hover { background: #1e4270; }"
            "QPushButton:pressed { background: #0f2742; }"
        )

        layout = QVBoxLayout(overlay)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        title = QLabel("Driver is SLEEPY", overlay)
        title.setObjectName("sleepTitle")
        subtitle = QLabel("Continue to Destination or Park Nearby?", overlay)
        subtitle.setObjectName("sleepSubtitle")
        countdown = QLabel("", overlay)
        countdown.setObjectName("sleepCountdown")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(countdown)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 6, 0, 0)
        button_row.setSpacing(10)
        continue_button = QPushButton("Continue to Destination", overlay)
        park_button = QPushButton("Park Nearby", overlay)
        continue_button.clicked.connect(self._on_sleep_decision_continue)
        park_button.clicked.connect(self._on_sleep_decision_park)
        button_row.addWidget(continue_button, stretch=1)
        button_row.addWidget(park_button, stretch=1)
        layout.addLayout(button_row)

        self._sleep_decision_overlay = overlay
        self._sleep_decision_countdown_label = countdown
        self._position_sleep_decision_overlay()

    def _position_sleep_decision_overlay(self) -> None:
        if self._sleep_decision_overlay is None:
            return
        try:
            parent = self._sleep_decision_overlay.parentWidget()
            if parent is None:
                return
            rect = parent.rect()
            width = min(620, max(420, int(rect.width() * 0.42)))
            height = 190
            x = int((rect.width() - width) / 2)
            y = int((rect.height() - height) / 2)
            self._sleep_decision_overlay.setGeometry(x, y, width, height)
            self._sleep_decision_overlay.raise_()
        except Exception:
            return

    def _show_sleep_decision_overlay(self) -> None:
        if self._sleep_decision_overlay is None:
            return
        self._sleep_decision_expires_at = time.time() + 5.0
        self._sleep_decision_visible = True
        self._position_sleep_decision_overlay()
        self._sleep_decision_overlay.setVisible(True)
        self._sleep_decision_overlay.raise_()

    def _hide_sleep_decision_overlay(self) -> None:
        if self._sleep_decision_overlay is None:
            return
        self._sleep_decision_visible = False
        self._sleep_decision_expires_at = None
        self._sleep_decision_overlay.setVisible(False)

    def _update_sleep_decision_overlay(self) -> None:
        if not self._sleep_decision_visible or self._sleep_decision_overlay is None:
            return
        if self._sleep_decision_expires_at is None:
            return

        now = time.time()
        remaining = max(0, int(math.ceil(self._sleep_decision_expires_at - now)))
        if self._sleep_decision_countdown_label is not None:
            self._sleep_decision_countdown_label.setText(f"Auto-parking in {remaining}s")

        if now >= self._sleep_decision_expires_at:
            self._hide_sleep_decision_overlay()
            self._append_decision("Sleepy decision timeout -> Park Nearby")
            self._execute_sleepy_parking(reason="timeout")

    def _on_sleep_decision_continue(self) -> None:
        self._hide_sleep_decision_overlay()
        self._append_decision("Sleepy decision: Continue to Destination")
        destination = self.controller.get_route_destination()
        if destination is None:
            self._append_decision("No destination set -> Park Nearby")
            self._execute_sleepy_parking(reason="no-destination")
            return

        if self.controller.is_route_following():
            self.autopilot_enabled = True
            self._set_status("Continuing to destination", "info")
            return

        if not self.controller.has_route_plan():
            planned, message = self.controller.plan_route_to_destination()
            if not planned:
                self._append_alert(message)
                self._set_status("Route planning failed", "bad")
                return

        started, message = self.controller.start_route_following(target_speed_kmh=30.0, force_replan=True)
        if started:
            self.autopilot_enabled = True
            self._append_decision(message)
            self._set_status("Autopilot route ON", "info")
        else:
            self.autopilot_enabled = False
            self._append_alert(message)
            self._set_status("Autopilot route failed", "bad")

    def _on_sleep_decision_park(self) -> None:
        self._hide_sleep_decision_overlay()
        self._append_decision("Sleepy decision: Park Nearby")
        self._execute_sleepy_parking(reason="user")

    def _execute_sleepy_parking(self, *, reason: str) -> None:
        if self.controller.vehicle is None:
            return

        try:
            already_shoulder = self.controller.is_in_shoulder_or_parking_lane()
        except Exception:
            already_shoulder = False

        if already_shoulder:
            message = "Already on shoulder/parking lane; skipping Park Nearby"
            self._append_decision(f"Sleepy parking: {message}")
            self._set_status(message, "bad")
            LOGGER.info("[sleepy-park] skip reason=%s already_on_shoulder=True", str(reason))
            return

        try:
            current_dest = self.controller.get_route_destination()
        except Exception:
            current_dest = None

        LOGGER.info(
            "[sleepy-park] trigger reason=%s had_destination=%s was_route_following=%s",
            str(reason),
            current_dest is not None,
            bool(self.controller.is_route_following()),
        )

        # Phase 1: first get to a driving lane whose RIGHT neighbor is shoulder/parking.
        # This avoids sharp merges into the shoulder when auto-parking engages.
        try:
            positioned = self.controller.is_positioned_for_right_shoulder_parking()
        except Exception:
            positioned = True

        if not positioned:
            self._sleep_right_shift_pending = True
            self._sleep_right_shift_started_at = time.time()
            self._sleep_right_shift_last_request_at = 0.0
            self._sleep_right_shift_rightmost_since = None
            self._append_decision("Sleepy parking: shifting right until shoulder is on the right")
            self._set_status("Sleepy: shifting right (seek shoulder)", "bad")
            LOGGER.info("[sleepy-park] phase=shift-right pending=True")
            return

        # Phase 2: already rightmost -> start parking route immediately.
        self._start_sleepy_parking_route(reason=reason)

    def _start_sleepy_parking_route(self, *, reason: str) -> None:
        if self.controller.vehicle is None:
            return

        try:
            already_shoulder = self.controller.is_in_shoulder_or_parking_lane()
        except Exception:
            already_shoulder = False

        if already_shoulder:
            message = "Already on shoulder/parking lane; no parking route needed"
            self._append_decision(f"Sleepy parking: {message}")
            self._set_status(message, "bad")
            LOGGER.info("[sleepy-park] route-skip reason=%s already_on_shoulder=True", str(reason))
            return

        # Stop any TrafficManager autopilot that may have been used for lane positioning.
        try:
            self.controller.set_autopilot_enabled(False)
        except Exception:
            pass

        try:
            self.autopilot_enabled = False
        except Exception:
            pass
        try:
            if self.controller.is_route_following():
                self.controller.stop_route_following(reason="sleepy-park")
        except Exception:
            pass

        ok, message = self.controller.start_nearest_shoulder_lane_parking_via_route(
            min_consecutive_waypoints=4,
            stop_ahead_waypoints=10,
            waypoint_spacing_m=2.0,
            scan_steps=45,
            target_speed_kmh=15.0,
            reason=f"sleepy:{reason}",
        )
        if ok:
            self.autopilot_enabled = True
            self._append_decision(message)
            self._set_status("Parking nearby shoulder", "bad")
            LOGGER.info("[sleepy-park] started %s", str(message))
            return

        self._append_alert(f"Parking search failed ({reason}): {message}")
        LOGGER.warning("[sleepy-park] failed reason=%s error=%s", str(reason), str(message))
        try:
            self.controller.request_pull_over()
            self._append_decision("Fallback: pull-over requested")
            self._set_status("Pull-over (no shoulder spot)", "bad")
        except Exception:
            pass

    def _run_startup_sound_check(self) -> None:
        if self._startup_sound_checked:
            return
        self._startup_sound_checked = True
        emitted = self.alerts.beep(message="Startup alert sound check", level="sleepy")
        if emitted:
            self._append_decision("Startup sound check OK")
        else:
            self._append_alert("Startup sound check failed")

    def _on_traffic_toggled(self, enabled: bool) -> None:
        self._toggle_traffic_process(enabled)

    def _toggle_detection_backend(self) -> None:
        """Switch between Haar Cascade and MediaPipe detection backends."""
        self._use_mediapipe = not self._use_mediapipe
        backend_name = "MEDIAPIPE" if self._use_mediapipe else "HAAR"

        if self._use_mediapipe:
            try:
                from .mediapipe_system import MediaPipeDrowsinessSystem
                if self._mediapipe_system is None:
                    self._mediapipe_system = MediaPipeDrowsinessSystem()
                self.backend_button.setText(f"Detection: {backend_name}")
                self._append_decision(f"Switched to {backend_name} detection")
            except ImportError as exc:
                self._use_mediapipe = False
                self._append_alert(f"MediaPipe unavailable: {exc}")
                self.backend_button.setText("Detection: HAAR")
        else:
            self._mediapipe_system = None
            self.backend_button.setText(f"Detection: {backend_name}")
            self._append_decision(f"Switched to {backend_name} detection")

    def _on_map_destination_selected(self, x_world: float, y_world: float) -> None:
        was_navigating = self.autopilot_enabled or self.controller.is_route_following()
        if was_navigating:
            self.controller.stop_route_following(reason="rerouting")
            self.controller.set_autopilot_enabled(False)
            if carla is not None and self.controller.vehicle is not None:
                try:
                    # Prevent stale throttle from the previous route while a new destination is being planned.
                    self.controller.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.35, steer=0.0, hand_brake=False))
                except Exception:
                    pass

        z_world = 0.0
        if self.controller.vehicle is not None:
            try:
                z_world = float(self.controller.vehicle.get_location().z)
            except Exception:
                z_world = 0.0
        ok, message = self.controller.set_destination_location(x_world, y_world, z_world)
        if ok:
            self._append_decision(message)
            if was_navigating:
                planned, plan_message = self.controller.plan_route_to_destination()
                if not planned:
                    self.autopilot_enabled = False
                    self._append_alert(plan_message)
                    self._set_status("Reroute planning failed", "warn")
                    return

                started, start_message = self.controller.start_route_following(target_speed_kmh=30.0)
                if started:
                    self.autopilot_enabled = True
                    self._append_decision("Destination changed during autopilot -> rerouting")
                    self._append_decision(start_message)
                    self._set_status("Rerouting to new destination", "info")
                else:
                    self.autopilot_enabled = False
                    self._append_alert(start_message)
                    self._set_status("Reroute start failed", "warn")
            else:
                self._set_status("Destination selected", "info")
        else:
            self._append_alert(message)
            self._set_status("Destination selection failed", "warn")

    def _on_map_plan_route(self) -> None:
        ok, message = self.controller.plan_route_to_destination()
        if ok:
            self._append_decision(message)
            self._set_status("Route planned", "good")
        else:
            self._append_alert(message)
            self._set_status("Route planning failed", "warn")

    def _on_map_start_route(self) -> None:
        self.autopilot_enabled = False
        self.controller.set_autopilot_enabled(False)
        ok, message = self.controller.start_route_following(force_replan=True)
        if ok:
            self._append_decision(message)
            self._set_status("Route following started", "info")
        else:
            self._append_alert(message)
            self._set_status("Route following failed", "warn")

    def _on_map_stop_route(self) -> None:
        self.controller.stop_route_following(reason="stopped")
        self._append_decision("Route following stopped")
        self._set_status("Route following stopped", "neutral")

    def _on_map_clear_destination(self) -> None:
        self.controller.clear_destination()
        self._append_decision("Destination cleared")
        self._set_status("Destination cleared", "neutral")

    def _show_destination_required_popup(self) -> None:
        if QMessageBox is not None:
            try:
                QMessageBox.warning(self, "Destination Required", "Please enter Destination")
                return
            except Exception:
                pass
        self._append_alert("Please enter Destination")

    def _start_navigation_autopilot(self) -> None:
        destination = self.controller.get_route_destination()
        if destination is None:
            self._show_destination_required_popup()
            self._set_status("Destination required", "warn")
            return

        # If the driver took over manually, the previously planned route may no longer be
        # valid from the current vehicle position. Force a fresh plan when enabling
        # autopilot so the vehicle can recover and get back onto the road.
        started, message = self.controller.start_route_following(force_replan=True)
        if started:
            self.autopilot_enabled = True
            self._append_decision(message)
            self._set_status("Autopilot route ON", "info")
        else:
            self.autopilot_enabled = False
            self._append_alert(message)
            self._set_status("Autopilot route failed", "bad")

    def _stop_navigation_autopilot(self) -> None:
        self.controller.stop_route_following(reason="stopped")
        self.controller.set_autopilot_enabled(False)
        self.autopilot_enabled = False
        self._set_status("Autopilot OFF", "neutral")

    def _tick_client_fps(self) -> float:
        now = time.perf_counter()
        if self._last_tick_time is None:
            self._last_tick_time = now
            return 0.0
        dt = max(1e-6, now - self._last_tick_time)
        self._last_tick_time = now
        fps = 1.0 / dt
        if self._client_fps_smoothed <= 0.0:
            self._client_fps_smoothed = fps
        else:
            self._client_fps_smoothed = self._client_fps_smoothed * 0.85 + fps * 0.15
        return self._client_fps_smoothed

    def _get_server_fps(self) -> float:
        if self.controller.world is None:
            return 0.0
        try:
            snapshot = self.controller.world.get_snapshot()
            delta = float(snapshot.timestamp.delta_seconds)
            return 0.0 if delta <= 0.0 else 1.0 / delta
        except Exception:
            return 0.0

    def _set_status(self, message: str, severity: str = "neutral") -> None:
        self._status_message = str(message)
        self._status_severity = str(severity)
        self._status_updated_at = time.time()
        self._status_source = "driver" if str(message).startswith("Driver state:") else "app"
        self.global_status.setText(message)
        _set_badge(self.global_status, message, severity)

    def _maybe_set_driver_state_status(self, state: DrowsinessState) -> None:
        """Only show driver-state in the global status when it won't clobber an active action status."""
        now = time.time()
        # If an app/action status was set very recently, keep it visible.
        if self._status_source == "app" and (now - float(self._status_updated_at)) < 2.0:
            return

        if state == DrowsinessState.ALERT:
            self._set_status("Driver state: ALERT", "good")
        elif state == DrowsinessState.DROWSY:
            self._set_status("Driver state: DROWSY", "warn")
        else:
            self._set_status("Driver state: SLEEP", "bad")

    def _append_alert(self, message: str) -> None:
        stamped = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        self.alert_log.append(stamped)

    def _append_decision(self, message: str) -> None:
        stamped = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        self.decision_log.append(stamped)

    def _set_vehicle_lights(self) -> None:
        if carla is None or self.controller.vehicle is None:
            return
        try:
            self.controller.vehicle.set_light_state(carla.VehicleLightState(self._lights_state))
        except Exception:
            pass

    def _toggle_light_cycle(self) -> None:
        if carla is None:
            return
        position = int(carla.VehicleLightState.Position)
        low_beam = int(carla.VehicleLightState.LowBeam)
        fog = int(carla.VehicleLightState.Fog)
        if not self._lights_state & position:
            self._lights_state |= position
            self._set_status("Position lights", "info")
        elif not self._lights_state & low_beam:
            self._lights_state |= low_beam
            self._set_status("Low beam lights", "info")
        elif not self._lights_state & fog:
            self._lights_state |= fog
            self._set_status("Fog lights", "warn")
        else:
            self._lights_state &= ~position
            self._lights_state &= ~low_beam
            self._lights_state &= ~fog
            self._set_status("Lights off", "neutral")
        self._set_vehicle_lights()

    def _toggle_traffic_process(self, enabled: bool) -> None:
        if not enabled:
            self._stop_traffic_process()
            self._cleanup_generated_traffic()
            self.dashboard_tab.set_traffic_running(False)
            self._set_status("Traffic generator stopped", "neutral")
            self._append_decision("Traffic OFF -> cleaned up generated actors")
            return

        if self._traffic_process is not None and self._traffic_process.poll() is None:
            self.dashboard_tab.set_traffic_running(True)
            self._set_status("Traffic generator already running", "info")
            return

        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "PythonAPI" / "examples" / "generate_traffic.py"
        if not script_path.exists():
            self._set_status("generate_traffic.py not found", "bad")
            self.dashboard_tab.set_traffic_running(False)
            return

        host = self.controller.connection.host if self.controller is not None else "127.0.0.1"
        port = str(self.controller.connection.port if self.controller is not None else 2000)
        command = [sys.executable, str(script_path), "--host", host, "--port", port]
        try:
            self._traffic_process = subprocess.Popen(command, cwd=str(repo_root))
            self.dashboard_tab.set_traffic_running(True)
            self._set_status("Traffic generator started", "good")
            self._append_decision("Traffic ON -> started generate_traffic.py")
        except Exception as exc:
            self._set_status(f"Traffic start failed: {exc}", "bad")
            self.dashboard_tab.set_traffic_running(False)

    def _stop_traffic_process(self) -> None:
        if self._traffic_process is None:
            return
        if self._traffic_process.poll() is not None:
            self._traffic_process = None
            return
        try:
            self._traffic_process.send_signal(signal.SIGINT)
            self._traffic_process.wait(timeout=3.0)
        except Exception:
            try:
                self._traffic_process.terminate()
                self._traffic_process.wait(timeout=2.0)
            except Exception:
                try:
                    self._traffic_process.kill()
                except Exception:
                    pass
        self._traffic_process = None

    def _cleanup_generated_traffic(self) -> None:
        if carla is None or self.controller.world is None or self.controller.client is None:
            return
        try:
            actors = self.controller.world.get_actors()
            destroy_ids: list[int] = []

            for walker_controller in list(actors.filter("controller.ai.walker")):
                try:
                    walker_controller.stop()
                except Exception:
                    pass
                destroy_ids.append(walker_controller.id)

            for vehicle in actors.filter("vehicle.*"):
                role_name = vehicle.attributes.get("role_name", "") if hasattr(vehicle, "attributes") else ""
                if role_name == "autopilot":
                    destroy_ids.append(vehicle.id)

            for walker in actors.filter("walker.pedestrian.*"):
                destroy_ids.append(walker.id)

            if destroy_ids:
                unique_ids = sorted(set(destroy_ids))
                self.controller.client.apply_batch([carla.command.DestroyActor(actor_id) for actor_id in unique_ids])
        except Exception:
            pass

    def _next_weather(self, reverse: bool = False) -> None:
        presets = _find_weather_presets()
        if self.controller.world is None or not presets:
            return
        if not hasattr(self, "_weather_index"):
            self._weather_index = 0
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(presets)
        weather, weather_name = presets[self._weather_index]
        self.controller.world.set_weather(weather)
        self._append_decision(f"Weather changed to {weather_name}")

    def _set_vehicle_lights_from_control(self, brake: float, reverse: bool) -> None:
        if carla is None:
            return
        brake_mask = int(carla.VehicleLightState.Brake)
        reverse_mask = int(carla.VehicleLightState.Reverse)
        if brake > 0.0:
            self._lights_state |= brake_mask
        else:
            self._lights_state &= ~brake_mask
        if reverse:
            self._lights_state |= reverse_mask
        else:
            self._lights_state &= ~reverse_mask
        self._set_vehicle_lights()

    def _apply_manual_control(self) -> None:
        if self.controller.vehicle is None or carla is None or self.autopilot_enabled or self.controller.is_route_following():
            return

        throttle = 0.6 if (Qt.Key_W in self._pressed_keys or Qt.Key_Up in self._pressed_keys) else 0.0
        brake = 0.8 if (Qt.Key_S in self._pressed_keys or Qt.Key_Down in self._pressed_keys) else 0.0
        hand_brake = Qt.Key_Space in self._pressed_keys

        if Qt.Key_A in self._pressed_keys or Qt.Key_Left in self._pressed_keys:
            self.steer_cache = max(-0.8, self.steer_cache - 0.04)
        elif Qt.Key_D in self._pressed_keys or Qt.Key_Right in self._pressed_keys:
            self.steer_cache = min(0.8, self.steer_cache + 0.04)
        else:
            self.steer_cache = 0.0

        control = carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=self.steer_cache,
            hand_brake=hand_brake,
            reverse=self.reverse_enabled,
            manual_gear_shift=self.manual_gear_shift,
            gear=self.current_gear,
        )
        self._set_vehicle_lights_from_control(brake, self.reverse_enabled)
        self.controller.vehicle.apply_control(control)

    def _detect_drowsiness(self, frame_bgr) -> tuple[DrowsinessState, str, float, object]:
        if self._use_mediapipe and self._mediapipe_system is not None:
            frame_bgr, eye_aspect_ratio, state = self._mediapipe_system.process_frame(frame_bgr)
            status = "MEDIAPIPE"
            if state == DrowsinessState.DROWSY:
                self.alerts.beep(message="Driver is drowsy", level="drowsy")
            elif state == DrowsinessState.SLEEPY:
                self.alerts.beep(message="Driver is sleepy", level="sleepy")
            return state, status, eye_aspect_ratio, frame_bgr

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(90, 90))

        eye_aspect_ratio = 0.0
        status = "NO FACE"
        for (face_x, face_y, face_w, face_h) in faces[:1]:
            face_region_gray = gray[face_y : face_y + face_h, face_x : face_x + face_w]
            eye_boxes = self.eye_cascade.detectMultiScale(face_region_gray, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20))
            eye_aspect_ratio = self._estimate_eye_aspect_ratio((face_x, face_y, face_w, face_h), list(eye_boxes))
            status = "EYES DETECTED" if len(eye_boxes) else "NO EYES"
            for (eye_x, eye_y, eye_w, eye_h) in eye_boxes:
                cv2.rectangle(frame_bgr, (face_x + eye_x, face_y + eye_y), (face_x + eye_x + eye_w, face_y + eye_y + eye_h), (0, 255, 0), 2)
            cv2.rectangle(frame_bgr, (face_x, face_y), (face_x + face_w, face_y + face_h), (255, 0, 0), 2)
            break

        state = self.detector.update(eye_aspect_ratio=eye_aspect_ratio, timestamp=time.time())
        if state == DrowsinessState.DROWSY:
            self.alerts.beep(message="Driver is drowsy", level="drowsy")
        elif state == DrowsinessState.SLEEPY:
            self.alerts.beep(message="Driver is sleepy", level="sleepy")
        return state, status, eye_aspect_ratio, frame_bgr

    @staticmethod
    def _estimate_eye_aspect_ratio(face_box: tuple[int, int, int, int], eye_boxes: list[tuple[int, int, int, int]]) -> float:
        if not eye_boxes:
            return 0.0
        face_x, face_y, face_w, face_h = face_box
        del face_x, face_y
        eye_heights = sum(max(1, eye_h) for _, _, _, eye_h in eye_boxes)
        eye_widths = sum(max(1, eye_w) for _, _, eye_w, _ in eye_boxes)
        face_scale = max(1, face_w + face_h)
        return min(1.0, (eye_heights / max(1, eye_widths)) * (face_scale / max(1, face_scale)))

    def _update_trajectory(self) -> None:
        if self.controller.vehicle is None:
            return
        try:
            location = self.controller.vehicle.get_location()
            self._latest_trajectory.append((float(location.x), float(location.y)))
        except Exception:
            return

    def _build_snapshot(self, client_fps: float, server_fps: float, ear: float, state: DrowsinessState, status: str) -> DashboardSnapshot:
        speed_kmh = self.controller.get_speed_kmh() if self.controller.vehicle is not None else 0.0
        live_control = self.controller.vehicle.get_control() if self.controller.vehicle is not None else None
        lane_line, road_line, neighbors_line, lane_change_line = _get_carla_lane_info(self.controller)
        state_text = _state_text(state)
        if self.autopilot_enabled:
            mode_text = "AUTOPILOT"
        elif self._parking_state != "Manual":
            mode_text = "PARKING"
        else:
            mode_text = "MANUAL"

        telemetry_lines = [
            f"Driver state: {state_text}",
            f"Status: {status}",
            f"EAR: {ear:.3f}",
            f"Vehicle mode: {mode_text}",
            f"Speed: {speed_kmh:.1f} km/h",
            f"Gear: {self.current_gear}",
            f"Reverse: {'ON' if self.reverse_enabled else 'OFF'}",
            f"Steer: {live_control.steer:+.2f}" if live_control is not None else "Steer: +0.00",
            f"Throttle: {live_control.throttle:.2f}" if live_control is not None else "Throttle: 0.00",
            f"Brake: {live_control.brake:.2f}" if live_control is not None else "Brake: 0.00",
            lane_line,
            road_line,
            neighbors_line,
            lane_change_line,
        ]

        decision_lines = list(self.decision_log)
        alert_lines = list(self.alert_log)
        trajectory_points = list(self._latest_trajectory)
        waypoints = self._cached_waypoints
        drowsiness_score = 10 if state == DrowsinessState.ALERT else 65 if state == DrowsinessState.DROWSY else 95
        return DashboardSnapshot(
            driver_state=state_text,
            mode=mode_text,
            parking_state=self._parking_state,
            speed_kmh=speed_kmh,
            client_fps=client_fps,
            server_fps=server_fps,
            ear=ear,
            drowsiness_score=drowsiness_score,
            gear=self.current_gear,
            reverse_enabled=self.reverse_enabled,
            steer=live_control.steer if live_control is not None else 0.0,
            throttle=live_control.throttle if live_control is not None else 0.0,
            brake=live_control.brake if live_control is not None else 0.0,
            camera_label=self.controller.get_camera_view_name() if self.controller.vehicle is not None else "Camera",
            camera_mode=self.controller.get_camera_mode().upper() if self.controller.vehicle is not None else self.camera_mode.upper(),
            lane_hint=lane_line,
            alert_lines=alert_lines,
            decision_lines=decision_lines,
            telemetry_lines=telemetry_lines,
            trajectory_points=trajectory_points,
            waypoints=waypoints,
        )

    def _tick(self) -> None:
        if self.controller.world is not None and self.controller.vehicle is not None:
            self.controller.tick()

        client_fps = self._tick_client_fps()
        self._apply_manual_control()
        route_status = self.controller.run_route_step() if self.controller.is_route_following() else self.controller.get_route_status()
        if route_status != self._last_route_status:
            if route_status == "arrived":
                self.autopilot_enabled = False
                self.controller.set_autopilot_enabled(False)
                self._append_decision("Route destination reached")
                self._set_status("Route arrived", "good")
                self.controller.clear_destination()
                self._append_decision("Destination cleared")
                if QMessageBox is not None:
                    try:
                        popup = QMessageBox(self)
                        popup.setIcon(QMessageBox.Information)
                        popup.setWindowTitle("Route")
                        popup.setText("Destination reached")
                        popup.setStandardButtons(QMessageBox.Ok)
                        popup.setModal(False)
                        popup.show()
                        self._arrival_popup = popup
                    except Exception:
                        pass
            elif route_status == "parking-running":
                self._append_decision("Route handoff: following parking waypoints")
                self._set_status("Parking waypoint follower active", "info")
            elif route_status == "rerouted":
                self._append_decision("Route deviation detected -> rerouted to destination")
                self._set_status("Route rerouted", "info")
            elif route_status == "failed":
                self._append_alert("Route following failed")
                self._set_status("Route failed", "bad")
            elif route_status == "running":
                self._append_decision("Route following running")
            elif route_status == "planned":
                self._append_decision("Route planned")
            self._last_route_status = route_status

        ok, webcam_frame = self.capture.read()
        if not ok:
            webcam_frame = _build_placeholder_frame(420, 320, "WEBCAM DISCONNECTED", "Check camera index or permissions")
        self._latest_webcam_frame = webcam_frame

        state, status, ear, annotated_webcam = self._detect_drowsiness(webcam_frame.copy())
        self.ear_history.append(ear)

        # If sleepy parking is waiting on lane positioning, keep requesting right-lane shifts.
        if self._sleep_right_shift_pending and self.controller.vehicle is not None:
            self._tick_sleepy_right_shift()

        if state == DrowsinessState.SLEEPY and self.controller.vehicle is not None:
            self._update_sleep_decision_overlay()
            now = time.time()
            if self._sleep_detected_since is None:
                self._sleep_detected_since = now

            elapsed_sleep = now - self._sleep_detected_since
            if elapsed_sleep >= self._sleep_safety_delay_seconds:
                if not self._sleep_safety_attempted:
                    self._sleep_safety_attempted = True
                    if self.controller.get_route_destination() is None:
                        self._append_decision("Sleepy after 3s with no destination -> Park Nearby")
                        self._execute_sleepy_parking(reason="no-destination")
                    else:
                        self._append_decision("Sleepy after 3s -> awaiting driver decision")
                        self._set_status("Sleepy: choose Continue or Park", "bad")
                        self._show_sleep_decision_overlay()
            elif not self._sleep_safety_pending_logged:
                self._sleep_safety_pending_logged = True
                self._append_decision(
                    f"Sleep detected; decision popup will appear after {self._sleep_safety_delay_seconds:.0f}s continuous SLEEPY"
                )
        else:
            self._sleep_detected_since = None
            self._sleep_safety_attempted = False
            self._sleep_safety_pending_logged = False
            self._hide_sleep_decision_overlay()
            if self._sleep_right_shift_pending:
                self._sleep_right_shift_pending = False
                self._sleep_right_shift_started_at = None
                self._sleep_right_shift_last_request_at = 0.0
                self._sleep_right_shift_rightmost_since = None
                try:
                    self.controller.set_autopilot_enabled(False)
                except Exception:
                    pass

        if self.controller.vehicle is not None and state in {DrowsinessState.DROWSY, DrowsinessState.SLEEPY}:
            now = time.time()
            if (
                self._drowsiness_log_last_state != state
                or (now - float(self._drowsiness_log_last_at)) >= 2.0
            ):
                self._append_alert(f"[{datetime.now().strftime('%H:%M:%S')}] Drowsiness warning: {state.value}")
                self._drowsiness_log_last_at = now
                self._drowsiness_log_last_state = state
            self.controller.apply_drowsy_slowdown(target_speed_kmh=15.0, brake_strength=0.4)
        else:
            self._drowsiness_log_last_state = None

        if self.detector.should_pull_over and self.controller.vehicle is not None:
            self.controller.request_pull_over()
            self._append_decision("Pull-over requested by drowsiness detector")

        carla_frame = self.controller.get_latest_rgb_frame() if self.controller.vehicle is not None else None
        parking_frame = self.controller.get_latest_parking_frame() if self.controller.vehicle is not None else None

        # Get waypoints and draw overlay on CARLA frame
        self._cached_waypoints = (
            self.controller.get_waypoints_ahead(distance_ahead_meters=100.0, max_waypoints=20)
            if self.controller.vehicle is not None
            else []
        )
        carla_frame = self.controller.draw_waypoints_on_frame(carla_frame, self._cached_waypoints)

        self._latest_carla_frame = carla_frame
        self._update_trajectory()

        speed_kmh = self.controller.get_speed_kmh() if self.controller.vehicle is not None else 0.0
        self.speed_history.append(speed_kmh)
        live_control = self.controller.vehicle.get_control() if self.controller.vehicle is not None else None
        self.steer_history.append(live_control.steer if live_control is not None else 0.0)
        self.throttle_history.append(live_control.throttle if live_control is not None else 0.0)
        self.brake_history.append(live_control.brake if live_control is not None else 0.0)

        if self.autopilot_enabled:
            self._parking_state = "Autonomous Driving" if speed_kmh >= 1.0 else "Searching Parking"
        else:
            self._parking_state = "Manual"

        server_fps = self._get_server_fps()
        snapshot = self._build_snapshot(client_fps, server_fps, ear, state, status)

        if self.controller.vehicle is not None and carla_frame is None:
            carla_frame = _build_placeholder_frame(960, 540, "CARLA VIEW", "Waiting for RGB sensor")
        if self.controller.vehicle is not None and parking_frame is None:
            parking_frame = _build_placeholder_frame(960, 540, "PARKING VIEW", "Waiting for Cosmos front camera")

        self.dashboard_tab.set_frames(carla_frame, webcam_frame)
        self.dashboard_tab.set_snapshot(snapshot)
        self.monitoring_tab.set_frame(annotated_webcam)
        self.monitoring_tab.set_snapshot(snapshot, self.ear_history)
        self.analytics_tab.set_snapshot(snapshot, self.speed_history, self.steer_history, self.throttle_history, self.brake_history)
        self.parking_tab.set_frame(parking_frame)
        self.parking_tab.set_snapshot(snapshot)

        map_waypoints = self.controller.get_all_map_waypoints(resolution=2.0) if self.controller.world is not None else []
        vehicle_location = self.controller.vehicle.get_location() if self.controller.vehicle is not None else None
        vehicle_heading = self.controller.vehicle.get_transform().rotation.yaw if self.controller.vehicle is not None else 0.0
        map_name = self.controller.world.get_map().name if self.controller.world is not None else "Loaded map"
        route_points = self.controller.get_route_polyline()
        destination = self.controller.get_route_destination()
        self.dashboard_tab.set_minimap(
            map_waypoints=map_waypoints,
            vehicle_location=vehicle_location,
            route_points=route_points,
            destination=destination,
            heading_deg=vehicle_heading,
        )
        self.map_tab.set_map_data(
            map_waypoints,
            vehicle_location=vehicle_location,
            map_name=map_name,
            resolution=2.0,
            destination=destination,
            route_points=route_points,
            route_status=self.controller.get_route_status(),
        )

        self.header_label.setText("Driver Monitoring & Autonomous Safety Dashboard")
        self.subheader_label.setText("QTabWidget layout | real-time graphs | CARLA-integrated control and monitoring")

        self._maybe_set_driver_state_status(state)

        if self._latest_carla_frame is None and self.controller.vehicle is None:
            self._append_decision("Waiting for CARLA vehicle connection")

    def _tick_sleepy_right_shift(self) -> None:
        if self.controller.vehicle is None:
            self._sleep_right_shift_pending = False
            self._sleep_right_shift_started_at = None
            return

        started_at = self._sleep_right_shift_started_at or time.time()
        now = time.time()

        try:
            positioned = self.controller.is_positioned_for_right_shoulder_parking()
        except Exception:
            positioned = True

        if positioned:
            if self._sleep_right_shift_rightmost_since is None:
                self._sleep_right_shift_rightmost_since = now
                self._append_decision(
                    f"Sleepy parking: shoulder on right -> holding {self._sleep_right_shift_hold_s:.0f}s"
                )
                LOGGER.info(
                    "[sleepy-park] phase=shift-right rightmost=True holding_s=%.1f",
                    float(self._sleep_right_shift_hold_s),
                )

            held_s = float(now - float(self._sleep_right_shift_rightmost_since))
            if held_s >= float(self._sleep_right_shift_hold_s):
                self._sleep_right_shift_pending = False
                self._sleep_right_shift_started_at = None
                self._sleep_right_shift_last_request_at = 0.0
                self._sleep_right_shift_rightmost_since = None
                self._append_decision("Sleepy parking: shoulder-on-right hold complete -> selecting parking destination")
                LOGGER.info("[sleepy-park] phase=shift-right complete=True")
                self._start_sleepy_parking_route(reason="after-rightmost")
            return

        # Not rightmost: reset stability timer.
        self._sleep_right_shift_rightmost_since = None

        if now - started_at >= float(self._sleep_right_shift_timeout_s):
            self._sleep_right_shift_pending = False
            self._sleep_right_shift_started_at = None
            self._sleep_right_shift_last_request_at = 0.0
            self._sleep_right_shift_rightmost_since = None
            self._append_alert("Sleepy parking: could not reach rightmost lane in time; parking anyway")
            LOGGER.warning("[sleepy-park] phase=shift-right timeout=%.2fs", float(now - started_at))
            self._start_sleepy_parking_route(reason="rightmost-timeout")
            return

        # Request right-lane change periodically (TrafficManager needs repeated nudges on multi-lane roads).
        if now - float(self._sleep_right_shift_last_request_at) >= 0.9:
            self._sleep_right_shift_last_request_at = now
            ok, msg = self.controller.engage_sleep_safety_autopilot_right_lane()
            if ok:
                self._append_decision(f"Rightmost shift: {msg}")
                LOGGER.info("[sleepy-park] phase=shift-right request ok msg=%s", str(msg))
            else:
                self._append_alert(f"Rightmost shift failed: {msg}")
                LOGGER.warning("[sleepy-park] phase=shift-right request failed msg=%s", str(msg))

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.isAutoRepeat():
            return
        self._pressed_keys.add(event.key())
        self._on_key_pressed(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if event.isAutoRepeat():
            return
        self._pressed_keys.discard(event.key())

    def _on_key_pressed(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()
            return
        if event.key() == Qt.Key_F11:
            self._fullscreen = not self._fullscreen
            self.showFullScreen() if self._fullscreen else self.showNormal()
            return

        if self.controller.vehicle is None:
            return

        mods = event.modifiers()
        ctrl = bool(mods & Qt.ControlModifier)
        shift = bool(mods & Qt.ShiftModifier)

        if event.key() == Qt.Key_P:
            if ctrl:
                self._append_decision("Replay requested from keyboard")
            else:
                if self.autopilot_enabled or self.controller.is_route_following():
                    self._stop_navigation_autopilot()
                    self._sleep_safety_engaged = False
                else:
                    self._start_navigation_autopilot()
        elif event.key() == Qt.Key_Backspace:
            self._append_decision("Vehicle switch requested")
            vehicles = list(self.controller.world.get_actors().filter("vehicle.*")) if self.controller.world is not None else []
            if vehicles:
                vehicles.sort(key=lambda actor: actor.id)
                current_id = self.controller.vehicle.id
                current_index = 0
                for index, actor in enumerate(vehicles):
                    if actor.id == current_id:
                        current_index = index
                        break
                self.controller.vehicle = vehicles[(current_index + 1) % len(vehicles)]
                self.controller.start_rgb_preview(
                    width=self.controller._camera_width,
                    height=self.controller._camera_height,
                    transform_index=self.controller.get_camera_transform_index(),
                    sensor_mode=self.controller.get_camera_mode(),
                )
        elif event.key() == Qt.Key_Tab:
            if shift:
                self.controller.cycle_camera_transform(reverse=True)
            else:
                self.controller.cycle_camera_transform()
        elif event.key() == Qt.Key_Y:
            self.controller.cycle_to_secondary_view()
        elif Qt.Key_1 <= event.key() <= Qt.Key_9:
            sensor_slot = event.key() - Qt.Key_1
            labels_count = len(self.controller.get_camera_view_labels())
            if sensor_slot < labels_count:
                self.controller.set_camera_transform_index(sensor_slot)
        elif event.key() == Qt.Key_Q:
            self.reverse_enabled = not self.reverse_enabled
            self._append_decision(f"Reverse mode {'enabled' if self.reverse_enabled else 'disabled'}")
        elif event.key() == Qt.Key_M:
            self.manual_gear_shift = not self.manual_gear_shift
        elif event.key() == Qt.Key_Comma and self.manual_gear_shift:
            self.current_gear -= 1
        elif event.key() == Qt.Key_Period and self.manual_gear_shift:
            self.current_gear += 1
        elif event.key() == Qt.Key_C:
            self._next_weather(reverse=shift)
        elif event.key() == Qt.Key_L:
            if ctrl and carla is not None:
                self._lights_state ^= int(carla.VehicleLightState.Special1)
            elif shift and carla is not None:
                self._lights_state ^= int(carla.VehicleLightState.HighBeam)
            else:
                self._toggle_light_cycle()
        elif event.key() == Qt.Key_T:
            self._append_decision("Telemetry shortcut pressed")
        elif event.key() == Qt.Key_H or event.key() == Qt.Key_Slash:
            self.show_help = not self.show_help
        elif event.key() == Qt.Key_F1:
            self.show_hud = not self.show_hud
        elif event.key() == Qt.Key_O:
            self._append_decision("Door toggle requested")
        elif event.key() == Qt.Key_G:
            self._append_decision("Radar toggle requested")

    def _key_down(self, key_code: int) -> bool:
        return key_code in self._pressed_keys

    def closeEvent(self, event) -> None:  # noqa: N802
        self.timer.stop()
        self._stop_traffic_process()
        self._cleanup_generated_traffic()
        if self.capture is not None:
            self.capture.release()
        self.controller.destroy()
        cv2.destroyAllWindows()
        event.accept()


def _ensure_pyqt_available() -> None:
    if QApplication is None:
        raise RuntimeError("PyQt5 is required for dashboard mode. Install PyQt5 to continue.")
    if pg is None:
        raise RuntimeError("pyqtgraph is required for dashboard mode. Install pyqtgraph to continue.")


def _configure_qt_runtime() -> None:
    if QLibraryInfo is None:
        return
    try:
        plugins_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)
        if plugins_path:
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugins_path
    except Exception:
        return


def _get_or_create_app():
    global _QT_APP
    _ensure_pyqt_available()
    _configure_qt_runtime()
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _QT_APP = app
    return app


def _run_qt_window(window) -> None:
    app = _get_or_create_app()
    owns_app = False
    window.show()
    if owns_app:
        app.exec_()
    else:
        app.exec_()


def run_camera_qt(
    camera_index: int,
    carla_controller: CarlaSafetyController,
    carla_view_width: int,
    carla_view_height: int,
    carla_camera_mode: str,
    carla_view_index: int,
    use_mediapipe: bool = False,
) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is not available. Install opencv-python to use camera mode.")
    _get_or_create_app()
    window = TabbedDashboardWindow(
        controller=carla_controller,
        camera_index=camera_index,
        carla_view_width=carla_view_width,
        carla_view_height=carla_view_height,
        carla_camera_mode=carla_camera_mode,
        carla_view_index=carla_view_index,
        use_mediapipe=use_mediapipe,
    )
    try:
        _run_qt_window(window)
    finally:
        try:
            carla_controller.destroy()
        except Exception:
            pass

