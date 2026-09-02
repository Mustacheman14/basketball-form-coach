"""Interactive test harness for the full multi-angle assessment session:
webcam feed with skeleton overlay, the current angle prompt, and reps-done
count, advancing through front -> strong-side -> guide-side -> back
automatically as each angle's 5 reps are completed.

Controls:
  q - quit
  m - manually count the current rep (if auto-detection missed it)
  d - discard the current rep buffer (if auto-detection false-triggered)
  n - skip to the next angle (escape hatch if a angle isn't converging)
"""

import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from core.pose_estimation import MODEL_PATH, draw_skeleton
from core.session import AssessmentSession


def run():
    side = input("Shooting hand - type 'left' or 'right': ").strip().lower()
    if side not in ("left", "right"):
        raise ValueError("Shooting hand must be 'left' or 'right'.")

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    session = AssessmentSession(shooting_side=side)

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
            if result.pose_landmarks and not session.is_complete:
                landmarks = result.pose_landmarks[0]
                draw_skeleton(frame, landmarks)
                state = session.update(landmarks)

            if session.is_complete:
                cv2.putText(frame, "SESSION COMPLETE", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"angle: {session.current_angle} ({state})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, session.current_prompt, (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"reps: {session.reps_done_this_angle}/{session.reps_per_angle}",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(frame, "q=quit  m=manual count  d=discard  n=skip angle", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Assessment Session Test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                session.manual_count_rep()
            elif key == ord("d"):
                session.discard_current_rep()
            elif key == ord("n"):
                session.skip_to_next_angle()

    cap.release()
    cv2.destroyAllWindows()

    print("\nFinal results:")
    for angle, reps in session.results.items():
        print(f"  {angle}: {len(reps)} reps captured")


if __name__ == "__main__":
    run()
