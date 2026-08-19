"""Interactive test harness: live webcam feed with skeleton overlay, plus
the rep-detection state machine's current state and rep count drawn on
screen. Run this yourself and take a few shots to check whether reps get
counted correctly -- this can't be validated by an automated frame test,
only by watching it against a real shooting motion.

Controls:
  q - quit
  m - manually count the current rep (if auto-detection missed it)
  d - discard the current rep buffer (if auto-detection false-triggered)
"""

import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from core.pose_estimation import MODEL_PATH, draw_skeleton
from core.rep_detection import RepCounter


def run():
    side = input("Shooting hand - type 'left' or 'right': ").strip().lower()
    if side not in ("left", "right"):
        raise ValueError("Shooting hand must be 'left' or 'right'.")

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    counter = RepCounter(shooting_side=side)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

            state = "no_pose"
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                draw_skeleton(frame, landmarks)
                state = counter.update(landmarks)

            cv2.putText(frame, f"state: {state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"reps: {counter.rep_count}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, "q=quit  m=manual count  d=discard", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Rep Detection Test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                counter.manual_count_rep()
            elif key == ord("d"):
                counter.discard_current_rep()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
