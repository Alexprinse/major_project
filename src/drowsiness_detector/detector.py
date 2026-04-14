from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time

from .config import DrowsinessConfig


class DrowsinessState(str, Enum):
    ALERT = "alert"
    DROWSY = "drowsy"
    SLEEPY = "sleepy"


@dataclass(slots=True)
class DetectorSample:
    eye_aspect_ratio: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class DrowsinessDetector:
    """Simple stateful detector scaffold.

    Replace ``score_sample`` with a real vision pipeline when integrating an
    eye landmark detector, CNN model, or temporal classifier.
    """

    config: DrowsinessConfig = field(default_factory=DrowsinessConfig)
    _last_state: DrowsinessState = DrowsinessState.ALERT
    _drowsy_since: float | None = None
    _sleepy_since: float | None = None

    def score_sample(self, sample: DetectorSample) -> float:
        """Return a normalized drowsiness score in the range [0, 1]."""
        if sample.eye_aspect_ratio <= self.config.min_eye_aspect_ratio:
            return 1.0
        if sample.eye_aspect_ratio <= self.config.min_eye_aspect_ratio * 1.25:
            return 0.75
        if sample.eye_aspect_ratio <= self.config.min_eye_aspect_ratio * 1.5:
            return 0.45
        return 0.0

    def update(self, eye_aspect_ratio: float, timestamp: float | None = None) -> DrowsinessState:
        """Update the detector state from a single frame/sample."""
        current_time = time.time() if timestamp is None else timestamp
        sample = DetectorSample(eye_aspect_ratio=eye_aspect_ratio, timestamp=current_time)
        score = self.score_sample(sample)

        if score >= self.config.drowsy_score_threshold:
            if self._drowsy_since is None:
                self._drowsy_since = current_time
            elapsed = current_time - self._drowsy_since
            if elapsed >= self.config.sleep_window_seconds:
                self._sleepy_since = self._sleepy_since or self._drowsy_since
                self._last_state = DrowsinessState.SLEEPY
            elif elapsed >= self.config.drowsy_window_seconds:
                self._last_state = DrowsinessState.DROWSY
            else:
                self._last_state = DrowsinessState.ALERT
        else:
            self._drowsy_since = None
            self._sleepy_since = None
            self._last_state = DrowsinessState.ALERT

        return self._last_state

    @property
    def state(self) -> DrowsinessState:
        return self._last_state

    @property
    def should_pull_over(self) -> bool:
        return self.should_pull_over_at(time.time())

    def should_pull_over_at(self, current_time: float) -> bool:
        if self._sleepy_since is None:
            return False
        return (current_time - self._sleepy_since) >= self.config.pull_over_delay_seconds