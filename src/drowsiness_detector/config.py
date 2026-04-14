from dataclasses import dataclass


@dataclass(slots=True)
class DrowsinessConfig:
    """Thresholds that drive the alert and pull-over logic."""

    blink_window_seconds: float = 3.0
    drowsy_window_seconds: float = 2.5
    sleep_window_seconds: float = 8.0
    alert_repeat_seconds: float = 2.5
    pull_over_delay_seconds: float = 5.0
    min_eye_aspect_ratio: float = 0.2
    drowsy_score_threshold: float = 0.6
