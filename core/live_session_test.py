"""Interactive test harness for the full multi-angle assessment session:
pre-session tips, webcam feed with skeleton overlay, angle prompts, rep
counting, and a shot-outcome prompt after every rep.

Controls (while the camera window is focused):
  q - quit
  m - manually count the current rep (if auto-detection missed it)
  d - discard the current rep buffer (if auto-detection false-triggered)
  n - skip to the next angle (escape hatch if an angle isn't converging)
  r - remove the most recently counted rep (undo a double count)
"""

import ctypes
import json
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from core.pose_estimation import MODEL_PATH, draw_skeleton
from core.session import AssessmentSession, PRE_SESSION_TIPS

FUNDAMENTALS_PATH = "data/fundamentals.json"

MISS_TYPES = [
    "Hit the front rim, bounced out",
    "Hit the back rim, bounced out",
    "Hit the left side of the rim, bounced out",
    "Hit the right side of the rim, bounced out",
    "Too low (short)",
    "Too low and right",
    "Too low and left",
    "Too right",
    "Too left",
    "Too high (long)",
    "Too high and left",
    "Too high and right",
    "Airball - too low",
    "Airball - too high",
    "Airball - too right",
    "Airball - too left",
]


def show_pre_session_tips():
    with open(FUNDAMENTALS_PATH) as f:
        fundamentals = json.load(f)

    print("\n=== Shooting Fundamentals ===")
    for cue in fundamentals["cues"]:
        print(f"- {cue['title']}: {cue['cue']}")

    print("\n=== Before You Start ===")
    for line in PRE_SESSION_TIPS:
        print(line)

    input("\nPress Enter when you're ready to start...\n")


def prompt_outcome(rep_record):
    print(f"\n--- Rep {rep_record['rep_number']} ({rep_record['angle']}) ---")
    made = input("Made it? (y/n): ").strip().lower()
    if made == "y":
        rep_record["outcome"] = {"made": True}
        return

    print("Miss type:")
    for i, label in enumerate(MISS_TYPES, start=1):
        print(f"  {i}. {label}")
    choice = input(f"Enter number (1-{len(MISS_TYPES)}, or blank to skip): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(MISS_TYPES):
        rep_record["outcome"] = {"made": False, "miss_type": MISS_TYPES[int(choice) - 1]}
    else:
        rep_record["outcome"] = {"made": False, "miss_type": None}


def get_screen_resolution():
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def draw_lines(frame, lines, start_y, font_scale=0.7, color=(255, 255, 0), line_height=32, x=10):
    for i, line in enumerate(lines):
        if line:
            cv2.putText(frame, line, (x, start_y + i * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)


def draw_debug_hud(frame, debug, screen_w):
    if not debug:
        return
    lines = [
        f"[debug] state={debug.get('state')} cooldown={debug.get('cooldown')}",
        f"wrist_y={debug.get('wrist_y', 0):.3f} smoothed={debug.get('smoothed_wrist_y', 0):.3f}",
        f"shoulder_y={debug.get('shoulder_y', 0):.3f} release_y={debug.get('release_y', 0):.3f}",
        f"torso_height={debug.get('torso_height', 0):.3f}",
    ]
    draw_lines(frame, lines, 30, font_scale=0.6, color=(0, 200, 255), line_height=28, x=screen_w - 500)


def run():
    side = input("Shooting hand - type 'left' or 'right': ").strip().lower()
    if side not in ("left", "right"):
        raise ValueError("Shooting hand must be 'left' or 'right'.")

    show_pre_session_tips()

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    pending_outcomes = []
    session = AssessmentSession(shooting_side=side, on_rep_complete=pending_outcomes.append)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    screen_w, screen_h = get_screen_resolution()
    window_name = "Assessment Session"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
            if not session.is_complete:
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    draw_skeleton(frame, landmarks)
                    state = session.update(landmarks)
                else:
                    # Tracking lost (e.g. stepped out of frame). Discard
                    # anything in progress rather than leaving stale state
                    # that could misfire once tracking resumes.
                    session.discard_current_rep()

            draw_debug_hud(frame, session.counter.debug, screen_w=frame.shape[1])

            # Upscale to the screen resolution so the fullscreen window is
            # actually filled, rather than showing the native 640x480 frame
            # in the corner with black space around it.
            frame = cv2.resize(frame, (screen_w, screen_h))

            if session.is_complete:
                draw_lines(frame, ["SESSION COMPLETE"], 50, font_scale=1.2, color=(0, 255, 0))
            else:
                draw_lines(frame, [f"angle: {session.current_angle} ({state})"], 50)
                draw_lines(frame, session.current_prompt_lines, 90)
                draw_lines(frame, [f"reps: {session.reps_done_this_angle}/{session.reps_per_angle}"], 170)

            draw_lines(
                frame,
                ["q=quit  m=manual count  d=discard  n=skip angle  r=undo last rep"],
                screen_h - 30, font_scale=0.6, color=(200, 200, 200), line_height=0,
            )

            if pending_outcomes:
                draw_lines(
                    frame,
                    ["REP COUNTED -- switch to the terminal to log make/miss"],
                    screen_h // 2, font_scale=1.0, color=(0, 0, 255),
                )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                session.manual_count_rep()
            elif key == ord("d"):
                session.discard_current_rep()
            elif key == ord("n"):
                session.skip_to_next_angle()
            elif key == ord("r"):
                removed = session.remove_last_rep()
                if removed:
                    print(f"Removed rep {removed['rep_number']} from {removed['angle']}")

            # Process outcome prompts only after the "REP COUNTED" frame has
            # actually been shown, so the terminal block isn't a surprise.
            while pending_outcomes:
                prompt_outcome(pending_outcomes.pop(0))

    cap.release()
    cv2.destroyAllWindows()

    print("\nFinal results:")
    for angle, reps in session.results.items():
        print(f"  {angle}: {len(reps)} reps captured")


if __name__ == "__main__":
    run()
