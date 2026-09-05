"""Streamlit web app: the real product interface, replacing the terminal
test harnesses now that headless OpenCV (required to avoid the Smart App
Control block -- see design-log.md decision #13) has no GUI window support.

Run with: streamlit run app/main.py
"""

import json
import threading
import time

import av
import cv2
import mediapipe as mp
import streamlit as st
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from streamlit_webrtc import webrtc_streamer

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


class LiveCoach:
    """Bridges the background WebRTC video thread and the Streamlit UI
    thread. All session/counter access goes through self.lock, since the
    two threads touch it concurrently."""

    def __init__(self, shooting_side):
        self.lock = threading.Lock()
        self.session = AssessmentSession(
            shooting_side=shooting_side, on_rep_complete=self._on_rep_complete
        )
        self.pending_outcomes = []
        self.start_time = time.time()
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def _on_rep_complete(self, rep_record):
        self.pending_outcomes.append(rep_record)

    def process_frame(self, frame_bgr):
        with self.lock:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - self.start_time) * 1000)
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            if not self.session.is_complete:
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    draw_skeleton(frame_bgr, landmarks)
                    self.session.update(landmarks)
                else:
                    self.session.discard_current_rep()
        return frame_bgr


def make_video_frame_callback(coach):
    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img = coach.process_frame(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    return callback


def load_fundamentals():
    with open(FUNDAMENTALS_PATH) as f:
        return json.load(f)["cues"]


def show_pre_session_tips():
    st.subheader("Shooting Fundamentals")
    for cue in load_fundamentals():
        st.markdown(f"**{cue['title']}** -- {cue['cue']}")

    st.subheader("Before You Start")
    for line in PRE_SESSION_TIPS:
        if line:
            st.markdown(line)


def render_outcome_form(coach):
    with coach.lock:
        if not coach.pending_outcomes:
            return
        rep_record = coach.pending_outcomes[0]

    st.warning(
        f"Rep {rep_record['rep_number']} ({rep_record['angle']}) counted -- log the outcome:"
    )
    with st.form(key=f"outcome_{rep_record['angle']}_{rep_record['rep_number']}"):
        made = st.radio("Made it?", ["Make", "Miss"], horizontal=True)
        miss_type = None
        if made == "Miss":
            miss_type = st.selectbox("Miss type", MISS_TYPES)
        submitted = st.form_submit_button("Log outcome")
        if submitted:
            with coach.lock:
                rep_record["outcome"] = (
                    {"made": True} if made == "Make"
                    else {"made": False, "miss_type": miss_type}
                )
                coach.pending_outcomes.pop(0)
            st.rerun()


def render_session_controls(coach):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Manual count"):
            with coach.lock:
                coach.session.manual_count_rep()
            st.rerun()
    with col2:
        if st.button("Discard current"):
            with coach.lock:
                coach.session.discard_current_rep()
            st.rerun()
    with col3:
        if st.button("Skip angle"):
            with coach.lock:
                coach.session.skip_to_next_angle()
            st.rerun()
    with col4:
        if st.button("Undo last rep"):
            with coach.lock:
                removed = coach.session.remove_last_rep()
            if removed:
                st.info(f"Removed rep {removed['rep_number']} from {removed['angle']}")
            st.rerun()


def main():
    st.set_page_config(page_title="Basketball Shooting Form Coach", layout="wide")
    st.title("Basketball Shooting Form Coach")

    if "coach" not in st.session_state:
        st.session_state.coach = None

    if st.session_state.coach is None:
        side = st.radio("Shooting hand", ["right", "left"], horizontal=True)
        show_pre_session_tips()
        if st.button("Start session"):
            st.session_state.coach = LiveCoach(shooting_side=side)
            st.rerun()
        return

    coach = st.session_state.coach

    webrtc_ctx = webrtc_streamer(
        key="assessment-session",
        video_frame_callback=make_video_frame_callback(coach),
        media_stream_constraints={"video": True, "audio": False},
    )

    render_outcome_form(coach)
    render_session_controls(coach)

    status = st.empty()
    progress = st.empty()

    while webrtc_ctx.state.playing:
        with coach.lock:
            session = coach.session
            if session.is_complete:
                complete = True
                results_summary = {angle: len(reps) for angle, reps in session.results.items()}
            else:
                complete = False
                angle = session.current_angle
                prompt_lines = list(session.current_prompt_lines)
                reps_done = session.reps_done_this_angle
                reps_total = session.reps_per_angle

        if complete:
            status.success("Session complete!")
            progress.write(results_summary)
            break

        status.markdown(f"### {angle}\n" + "\n".join(prompt_lines))
        progress.progress(reps_done / reps_total, text=f"{reps_done}/{reps_total} reps")
        time.sleep(0.3)


if __name__ == "__main__":
    main()
