from __future__ import annotations

import time

from drowsiness_detector.alerts import AlertController


def main() -> None:
    alerts = AlertController(drowsy_interval_seconds=0.5, sleepy_interval_seconds=0.2)
    print("Testing DROWSY beep (single, low intensity)...")
    ok_drowsy = alerts.beep(message="Sound test: drowsy", level="drowsy")
    print(f"DROWSY emitted: {ok_drowsy}")

    time.sleep(1.0)

    print("Testing SLEEPY beep (faster, higher intensity)...")
    ok_sleepy = alerts.beep(message="Sound test: sleepy", level="sleepy")
    print(f"SLEEPY emitted: {ok_sleepy}")

    print("If you heard nothing, install one of: canberra-gtk-play or paplay, or check system sound output.")


if __name__ == "__main__":
    main()
