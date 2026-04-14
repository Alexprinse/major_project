import unittest

from drowsiness_detector.config import DrowsinessConfig
from drowsiness_detector.detector import DrowsinessDetector, DrowsinessState


class DrowsinessDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DrowsinessConfig(
            drowsy_window_seconds=1.0,
            sleep_window_seconds=2.0,
            pull_over_delay_seconds=3.0,
            drowsy_score_threshold=0.6,
            min_eye_aspect_ratio=0.2,
        )
        self.detector = DrowsinessDetector(config=self.config)

    def test_stays_alert_when_eye_aspect_ratio_is_normal(self) -> None:
        state = self.detector.update(eye_aspect_ratio=0.34, timestamp=100.0)

        self.assertEqual(state, DrowsinessState.ALERT)
        self.assertEqual(self.detector.state, DrowsinessState.ALERT)
        self.assertFalse(self.detector.should_pull_over_at(100.0))

    def test_transitions_to_drowsy_on_low_eye_aspect_ratio(self) -> None:
        first_state = self.detector.update(eye_aspect_ratio=0.18, timestamp=100.0)
        second_state = self.detector.update(eye_aspect_ratio=0.18, timestamp=101.1)

        self.assertEqual(first_state, DrowsinessState.ALERT)
        self.assertEqual(second_state, DrowsinessState.DROWSY)
        self.assertEqual(self.detector.state, DrowsinessState.DROWSY)
        self.assertFalse(self.detector.should_pull_over_at(101.0))

    def test_transitions_to_sleepy_after_sustained_drowsiness(self) -> None:
        self.detector.update(eye_aspect_ratio=0.18, timestamp=100.0)
        state = self.detector.update(eye_aspect_ratio=0.18, timestamp=102.2)

        self.assertEqual(state, DrowsinessState.SLEEPY)
        self.assertEqual(self.detector.state, DrowsinessState.SLEEPY)
        self.assertFalse(self.detector.should_pull_over_at(102.9))
        self.assertTrue(self.detector.should_pull_over_at(103.0))

    def test_resets_to_alert_when_driver_recovers(self) -> None:
        self.detector.update(eye_aspect_ratio=0.18, timestamp=100.0)
        self.detector.update(eye_aspect_ratio=0.18, timestamp=102.2)
        state = self.detector.update(eye_aspect_ratio=0.34, timestamp=105.0)

        self.assertEqual(state, DrowsinessState.ALERT)
        self.assertEqual(self.detector.state, DrowsinessState.ALERT)
        self.assertFalse(self.detector.should_pull_over_at(105.0))


if __name__ == "__main__":
    unittest.main()
