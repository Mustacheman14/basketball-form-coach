"""Standalone smoke test: webcam feed with MediaPipe pose skeleton overlay.

Uses the MediaPipe Tasks API (mediapipe>=1.0 dropped the old mp.solutions
shortcut), so landmarks are drawn manually with OpenCV using the standard
33-point BlazePose body topology.

Run directly (`python core/pose_estimation.py`) to confirm the webcam and
pose model work before wiring this into the Streamlit app. Press 'q' to quit.
"""

import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_PATH = "models/pose_landmarker_lite.task"

# Standard BlazePose body connections (indices per the 33-point topology).
# Face landmarks (0-10) are excluded since they aren't needed for shooting form.
BODY_CONNECTIONS = [
    (11, 12),  # shoulders
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 23), (12, 24),  # torso sides
    (23, 24),  # hips
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
    (27, 29), (29, 31), (27, 31),  # left foot
    (28, 30), (30, 32), (28, 32),  # right foot
]


def draw_skeleton(frame, landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for start_idx, end_idx in BODY_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)

    for idx in {i for pair in BODY_CONNECTIONS for i in pair}:
        cv2.circle(frame, points[idx], 4, (0, 0, 255), -1)


def run():
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    start_time = time.time()

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.time() - start_time) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                draw_skeleton(frame, result.pose_landmarks[0])

            cv2.imshow("Pose Estimation Smoke Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
