"""Streamlit web app: the real product interface, replacing the terminal
test harnesses now that headless OpenCV (required to avoid the Smart App
Control block -- see design-log.md decision #13) has no GUI window support.

Run with: streamlit run app/main.py
"""

import json
import sys
import threading
import time
from pathlib import Path

import av
import cv2
import mediapipe as mp
import streamlit as st
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from streamlit_webrtc import webrtc_streamer

# `streamlit run` executes this file as a standalone script, so only its own
# directory (app/) is added to sys.path -- the project root has to be added
# manually for `core` to be importable, the same issue hit earlier with
# `python core/live_rep_test.py` vs `python -m core.live_rep_test`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.feedback import generate_report
from core.pose_estimation import MODEL_PATH, draw_skeleton
from core.session import ANGLE_SEQUENCE, AssessmentSession, PRE_SESSION_TIPS

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

CUSTOM_CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 1100px; }
h1 { font-weight: 700; }
div[data-testid="stMetricValue"] { font-size: 1.8rem; }
.stProgress > div > div { background-color: #ff6b35; }
</style>
"""


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

            self._draw_overlay(frame_bgr)
        return frame_bgr

    def _draw_overlay(self, frame_bgr):
        """Baked directly into the video pixels (not a separate Streamlit
        element) so the key info -- angle, prompt, rep count -- stays
        visible even when the video itself is fullscreened, which hides
        everything else on the page."""
        session = self.session
        if session.is_complete:
            cv2.putText(frame_bgr, "SESSION COMPLETE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            return

        y = 25
        angle_label = session.current_angle.replace("_", " ").title()
        cv2.putText(frame_bgr, angle_label, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += 25
        for line in session.current_prompt_lines:
            if line:
                cv2.putText(frame_bgr, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                y += 20
        y += 5
        cv2.putText(
            frame_bgr,
            f"reps: {session.reps_done_this_angle}/{session.reps_per_angle}",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )


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

    st.info(
        "This is an automated detector, not a referee -- it will sometimes miss a "
        "rep or double-count one. Use the buttons during the session to add or "
        "remove a rep yourself when that happens."
    )


def render_outcome_form(coach, rep_record):
    st.warning(
        f"Rep {rep_record['rep_number']} ({rep_record['angle'].replace('_', ' ')}) "
        f"counted -- log the outcome before the next shot:"
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
                if rep_record in coach.pending_outcomes:
                    coach.pending_outcomes.remove(rep_record)
            st.rerun(scope="fragment")


def render_session_controls(coach):
    st.caption(
        "Detection isn't perfect -- use these if it misses a shot or gets a count wrong."
    )
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("Manual count", help="Complete a shot the detector started tracking but never finished"):
            with coach.lock:
                coach.session.manual_count_rep()
            st.rerun(scope="fragment")
    with col2:
        if st.button("Force add rep", help="Add a rep even if the detector saw no motion at all"):
            with coach.lock:
                coach.session.force_add_rep()
            st.rerun(scope="fragment")
    with col3:
        if st.button("Discard current", help="Clear a false trigger the detector is mid-tracking"):
            with coach.lock:
                coach.session.discard_current_rep()
            st.rerun(scope="fragment")
    with col4:
        if st.button("Skip angle", help="Move on even without a full 5 reps"):
            with coach.lock:
                coach.session.skip_to_next_angle()
            st.rerun(scope="fragment")
    with col5:
        if st.button("Undo last rep", help="Remove the most recently counted rep"):
            with coach.lock:
                removed = coach.session.remove_last_rep()
            if removed:
                st.toast(f"Removed rep {removed['rep_number']} from {removed['angle']}")
            st.rerun(scope="fragment")


def render_report(results):
    report = generate_report(results)
    stats = report["stats"]

    st.subheader("Shooting")
    if stats["total"]:
        col1, col2, col3 = st.columns(3)
        col1.metric("Makes", stats["makes"])
        col2.metric("Misses", stats["misses"])
        col3.metric("Make %", f"{stats['make_pct']:.0f}%")
        if stats["miss_type_counts"]:
            st.caption("Miss breakdown:")
            for miss_type, count in sorted(stats["miss_type_counts"].items(), key=lambda kv: -kv[1]):
                st.markdown(f"- {miss_type}: {count}")
    else:
        st.caption("No shot outcomes were logged this session.")

    st.subheader("Form")
    if report["form_flags"]:
        st.caption(
            "These are first-pass, unverified thresholds -- treat them as things "
            "worth a second look, not a diagnosis."
        )
        for flag in report["form_flags"]:
            with st.container(border=True):
                st.markdown(f"**{flag['name']}**")
                st.caption(flag["cause"])
                st.markdown(f"Drill: {flag['drill']}")
    else:
        st.success("No form issues stood out from this session's data.")


def main():
    st.set_page_config(page_title="Basketball Shooting Form Coach", page_icon="🏀", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("🏀 Basketball Shooting Form Coach")

    if "coach" not in st.session_state:
        st.session_state.coach = None

    if st.session_state.coach is None:
        side = st.radio("Shooting hand", ["right", "left"], horizontal=True)
        show_pre_session_tips()
        if st.button("Start session", type="primary"):
            st.session_state.coach = LiveCoach(shooting_side=side)
            st.rerun()
        return

    coach = st.session_state.coach

    video_col, panel_col = st.columns([3, 2])
    with video_col:
        webrtc_ctx = webrtc_streamer(
            key="assessment-session",
            video_frame_callback=make_video_frame_callback(coach),
            media_stream_constraints={"video": True, "audio": False},
        )

    with panel_col:
        live_panel(coach, webrtc_ctx)


@st.fragment(run_every=0.4)
def live_panel(coach, webrtc_ctx):
    with coach.lock:
        session = coach.session
        debug = dict(session.counter.debug)
        pending_rep = coach.pending_outcomes[0] if coach.pending_outcomes else None
        complete = session.is_complete
        if not complete:
            angle = session.current_angle
            reps_done = session.reps_done_this_angle
            reps_total = session.reps_per_angle
            angle_number = ANGLE_SEQUENCE.index(angle) + 1

    if complete:
        st.success("Session complete!")
        with coach.lock:
            results_copy = {a: list(r) for a, r in session.results.items()}
        render_report(results_copy)
        return

    st.markdown(f"### Angle {angle_number}/{len(ANGLE_SEQUENCE)}: {angle.replace('_', ' ').title()}")
    st.progress(reps_done / reps_total, text=f"{reps_done}/{reps_total} reps this angle")

    if pending_rep:
        render_outcome_form(coach, pending_rep)
    else:
        render_session_controls(coach)

    with st.expander("Debug info (tuning detection, not needed for normal use)"):
        if debug:
            st.code(
                f"state={debug.get('state')} cooldown={debug.get('cooldown')}\n"
                f"wrist_y={debug.get('wrist_y', 0):.3f} smoothed={debug.get('smoothed_wrist_y', 0):.3f}\n"
                f"shoulder_y={debug.get('shoulder_y', 0):.3f} release_y={debug.get('release_y', 0):.3f}\n"
                f"torso_height={debug.get('torso_height', 0):.3f}"
            )
        else:
            st.caption("No frames processed yet.")

    if not webrtc_ctx.state.playing:
        st.warning("Camera not connected -- start the video stream above.")


if __name__ == "__main__":
    main()
