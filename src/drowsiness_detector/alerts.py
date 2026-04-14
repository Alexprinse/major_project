from __future__ import annotations

import sys
import time
import math
import shutil
import subprocess
import tempfile
import wave
import struct
from dataclasses import dataclass, field

try:
    import winsound  # type: ignore[attr-defined]
except Exception:
    winsound = None

try:
    from PyQt5.QtWidgets import QApplication  # type: ignore[import-not-found]
except Exception:
    QApplication = None


@dataclass
class AlertController:
    """Alert helper that emits a terminal bell or a text fallback."""

    min_interval_seconds: float = 2.5
    drowsy_interval_seconds: float = 2.5
    sleepy_interval_seconds: float = 0.8
    _last_alert_time: float = 0.0
    _last_alert_time_by_level: dict[str, float] = field(default_factory=dict)
    _tone_file_by_level: dict[str, str] = field(default_factory=dict)

    def _build_tone_file(self, level_key: str) -> str | None:
        frequency_hz = 1500 if level_key == "sleepy" else 900
        duration_ms = 190 if level_key == "sleepy" else 120
        amplitude = 0.9 if level_key == "sleepy" else 0.45
        sample_rate = 44100
        frame_count = int(sample_rate * duration_ms / 1000)
        if frame_count <= 0:
            return None

        try:
            tmp = tempfile.NamedTemporaryFile(prefix=f"drowsy_{level_key}_", suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()

            with wave.open(tmp_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)

                frames = bytearray()
                for i in range(frame_count):
                    t = i / float(sample_rate)
                    sample = amplitude * math.sin(2.0 * math.pi * frequency_hz * t)
                    sample_i16 = int(max(-1.0, min(1.0, sample)) * 32767.0)
                    frames.extend(struct.pack("<h", sample_i16))
                wav_file.writeframes(bytes(frames))
            return tmp_path
        except Exception:
            return None

    def _get_or_create_tone_file(self, level_key: str) -> str | None:
        existing = self._tone_file_by_level.get(level_key)
        if existing:
            return existing
        created = self._build_tone_file(level_key)
        if created is not None:
            self._tone_file_by_level[level_key] = created
        return created

    @staticmethod
    def _run_sound_command(command: list[str]) -> bool:
        try:
            subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            return False

    def _emit_single_beep(self, level_key: str) -> bool:
        # Windows native beep when available.
        if winsound is not None:
            try:
                if level_key == "sleepy":
                    winsound.Beep(1400, 180)
                else:
                    winsound.Beep(950, 110)
                return True
            except Exception:
                pass

        # Desktop audio fallback for Linux environments with verified command success.
        tone_file = self._get_or_create_tone_file(level_key)
        if tone_file is not None:
            for command in (
                ["aplay", "-q", tone_file],
                ["paplay", tone_file],
            ):
                if shutil.which(command[0]) is None:
                    continue
                if self._run_sound_command(command):
                    return True

        if shutil.which("canberra-gtk-play") is not None:
            if self._run_sound_command(["canberra-gtk-play", "-i", "bell"]):
                return True

        # Qt beep fallback when no direct audio backend worked.
        if QApplication is not None:
            try:
                app = QApplication.instance()
                if app is not None:
                    QApplication.beep()
                    return True
            except Exception:
                pass

        # Last fallback: terminal bell (often muted in IDE terminals).
        try:
            sys.stdout.write("\a")
            sys.stdout.flush()
            return True
        except Exception:
            return False

    def beep(self, message: str = "Driver drowsiness detected", level: str = "drowsy") -> bool:
        now = time.time()
        level_key = str(level).strip().lower() or "drowsy"
        if level_key == "sleepy":
            min_interval = self.sleepy_interval_seconds
            repeat_count = 1
        else:
            min_interval = self.drowsy_interval_seconds if self.drowsy_interval_seconds > 0.0 else self.min_interval_seconds
            repeat_count = 1

        last_time = self._last_alert_time_by_level.get(level_key, 0.0)
        if now - last_time < min_interval:
            return False

        self._last_alert_time_by_level[level_key] = now
        self._last_alert_time = now
        emitted = False
        for _ in range(repeat_count):
            emitted = self._emit_single_beep(level_key) or emitted

        print(f"[{level_key.upper()}] {message}")
        return emitted
