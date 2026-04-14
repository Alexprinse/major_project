from __future__ import annotations

from collections import deque
import time

import cv2
import mediapipe as mp
import numpy as np

from .detector import DrowsinessDetector, DrowsinessState


class MediaPipeDrowsinessSystem:
    def __init__(self) -> None:
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        self.detector = DrowsinessDetector()

        self.ear_history = deque(maxlen=5)

        self.blink_counter = 0
        self.blinks = 0
        self.blink_threshold = 0.2
        self.blink_frames = 3

        self.yawn_counter = 0
        self.yawns = 0
        self.yawn_event_times: deque[float] = deque()
        self.yawn_drowsy_count_threshold = 3
        self.yawn_event_window_seconds = 60.0
        self.yawn_threshold = 0.6
        self.yawn_frames = 15

        self.head_positions = deque(maxlen=10)
        self.nod_threshold = 15

    def _dist(self, p1: tuple[int, int], p2: tuple[int, int]) -> float:
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def _compute_ear(
        self,
        landmarks,
        eye_indices: list[int],
        width: int,
        height: int,
    ) -> tuple[float, list[tuple[int, int]]]:
        points: list[tuple[int, int]] = []
        for idx in eye_indices:
            lm = landmarks[idx]
            points.append((int(lm.x * width), int(lm.y * height)))

        p1, p2, p3, p4, p5, p6 = points
        ear = (self._dist(p2, p6) + self._dist(p3, p5)) / (2.0 * self._dist(p1, p4) + 1e-6)
        return ear, points

    def _compute_mar(self, landmarks, width: int, height: int) -> tuple[float, list[tuple[int, int]]]:
        def get_point(index: int) -> tuple[int, int]:
            lm = landmarks[index]
            return (int(lm.x * width), int(lm.y * height))

        p_top = get_point(13)
        p_bottom = get_point(14)
        p_left = get_point(78)
        p_right = get_point(308)

        vertical = np.linalg.norm(np.array(p_top) - np.array(p_bottom))
        horizontal = np.linalg.norm(np.array(p_left) - np.array(p_right))

        mar = float(vertical / (horizontal + 1e-6))
        return mar, [p_top, p_bottom, p_left, p_right]

    def _compute_head_drop(self, landmarks, width: int, height: int) -> int:
        del width
        nose = landmarks[1]
        chin = landmarks[152]

        nose_y = int(nose.y * height)
        chin_y = int(chin.y * height)
        vertical_dist = chin_y - nose_y
        return vertical_dist

    def process_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, DrowsinessState]:
        now = time.time()
        height, width, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return frame_bgr, 0.0, DrowsinessState.ALERT

        landmarks = results.multi_face_landmarks[0].landmark

        left_ear, left_points = self._compute_ear(landmarks, self.LEFT_EYE, width, height)
        right_ear, right_points = self._compute_ear(landmarks, self.RIGHT_EYE, width, height)
        ear = (left_ear + right_ear) / 2.0
        mar, mouth_points = self._compute_mar(landmarks, width, height)
        head_dist = self._compute_head_drop(landmarks, width, height)
        self.head_positions.append(head_dist)
        avg_head = float(np.mean(self.head_positions))
        nodding = False

        self.ear_history.append(ear)
        ear_smoothed = float(np.mean(self.ear_history))

        if ear_smoothed < self.blink_threshold:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.blink_frames:
                self.blinks += 1
            self.blink_counter = 0

        if mar > self.yawn_threshold:
            self.yawn_counter += 1
        else:
            if self.yawn_counter >= self.yawn_frames:
                self.yawns += 1
                self.yawn_event_times.append(now)
            self.yawn_counter = 0

        # Keep only recent yawn events so drowsy-by-yawn can recover to alert.
        while self.yawn_event_times and (now - self.yawn_event_times[0]) > self.yawn_event_window_seconds:
            self.yawn_event_times.popleft()

        recent_yawns = len(self.yawn_event_times)

        if len(self.head_positions) >= 5:
            nodding = head_dist < (avg_head - self.nod_threshold)

        state = self.detector.update(ear_smoothed, now)
        if recent_yawns >= self.yawn_drowsy_count_threshold and state == DrowsinessState.ALERT:
            state = DrowsinessState.DROWSY
        if nodding:
            state = DrowsinessState.SLEEPY

        for (x, y) in left_points + right_points:
            cv2.circle(frame_bgr, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in mouth_points:
            cv2.circle(frame_bgr, (x, y), 2, (255, 0, 0), -1)

        cv2.putText(
            frame_bgr,
            f"EAR: {ear_smoothed:.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame_bgr,
            f"Blinks: {self.blinks}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        color = (0, 255, 0)
        if state == DrowsinessState.DROWSY:
            color = (0, 255, 255)
        elif state == DrowsinessState.SLEEPY:
            color = (0, 0, 255)

        cv2.putText(
            frame_bgr,
            f"STATE: {state.value.upper()}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
        cv2.putText(
            frame_bgr,
            f"MAR: {mar:.2f}",
            (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
        )
        cv2.putText(
            frame_bgr,
            f"Yawns: {self.yawns} (recent: {recent_yawns})",
            (10, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            frame_bgr,
            f"Head: {head_dist:.1f}",
            (10, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
        )
        cv2.putText(
            frame_bgr,
            f"Nod: {'YES' if nodding else 'NO'}",
            (10, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255) if nodding else (0, 255, 0),
            2,
        )

        return frame_bgr, ear_smoothed, state
