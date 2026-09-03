"""Shot-cycle rep detection: segments a continuous stream of pose frames
into individual shot reps and counts them automatically.

Cycle signal is the shooting wrist's height relative to the shoulder,
scaled by the shooter's own torso height rather than a fixed body part
(MediaPipe's y increases downward, so "above" means a smaller y value):

    IDLE (wrist below shoulder)
      -> RISING (wrist crosses above shoulder)
      -> PEAKED (wrist rises at least RELEASE_HEIGHT_RATIO torso-heights
                 above the shoulder -- a release-height event)
      -> back to IDLE (wrist drops below shoulder again) => rep counted

An earlier version used the nose landmark as the "release height"
reference. That works from the front, but the face isn't visible to the
camera from the back angle (and only partially from the side angles) --
MediaPipe still returns a guessed nose position with low confidence in that
case, which made rep counting unreliable specifically on those angles.
Torso height (shoulder-to-hip distance) is used as the scale reference
instead, since both landmarks stay visible from all 4 session angles.

The raw wrist-height signal also jitters frame to frame, which without
smoothing can flicker across a threshold and register as a phantom rep or
miss a real one; a short rolling average absorbs that. A cooldown after
each completed rep guards against the following few frames -- arm still
settling from the follow-through -- immediately re-triggering a second
count.

Each completed rep returns the buffered per-frame metrics (from
angle_math.py) collected during that rep, for later aggregation across the
5 reps in a session. Auto-detection can still misfire, so
manual_count_rep() and discard_current_rep() exist as the override a user
can trigger from the UI.
"""

from collections import deque

from core import angle_math

STATE_IDLE = "idle"
STATE_RISING = "rising"
STATE_PEAKED = "peaked"

# A real shot cycle (rising -> peaked -> back to idle) takes at least a few
# hundred milliseconds. Without a minimum length, noise crossing the
# threshold during the descent can look like a full, extremely short cycle
# and get double-counted as a second rep.
MIN_REP_FRAMES = 10

# How far above the shoulder, in torso-heights, the wrist must rise to
# count as a release-height event.
RELEASE_HEIGHT_RATIO = 0.8

# Frames of rolling average applied to the wrist-height signal before it's
# compared against thresholds, to absorb single-frame tracking jitter.
SMOOTHING_WINDOW = 4

# Frames to ignore state transitions for immediately after a rep completes,
# so the arm settling from the follow-through can't trigger a second count.
COOLDOWN_FRAMES = 8


class RepCounter:
    def __init__(
        self,
        shooting_side,
        on_rep_complete=None,
        min_rep_frames=MIN_REP_FRAMES,
        release_height_ratio=RELEASE_HEIGHT_RATIO,
    ):
        self.shooting_side = shooting_side
        self.on_rep_complete = on_rep_complete
        self.min_rep_frames = min_rep_frames
        self.release_height_ratio = release_height_ratio
        self.state = STATE_IDLE
        self.rep_count = 0
        self._buffer = []
        self._wrist_y_window = deque(maxlen=SMOOTHING_WINDOW)
        self._cooldown = 0

    def reset(self):
        self.state = STATE_IDLE
        self.rep_count = 0
        self._buffer = []
        self._wrist_y_window.clear()
        self._cooldown = 0

    def update(self, landmarks):
        """Feed one frame's landmarks in. Returns the current state."""
        side = self.shooting_side
        wrist_y = landmarks[angle_math.LANDMARK[f"{side}_wrist"]].y
        shoulder_y = landmarks[angle_math.LANDMARK[f"{side}_shoulder"]].y
        hip_y = landmarks[angle_math.LANDMARK[f"{side}_hip"]].y

        self._wrist_y_window.append(wrist_y)
        smoothed_wrist_y = sum(self._wrist_y_window) / len(self._wrist_y_window)

        if self._cooldown > 0:
            self._cooldown -= 1
            return self.state

        torso_height = abs(hip_y - shoulder_y)
        release_y = shoulder_y - self.release_height_ratio * torso_height

        if self.state == STATE_IDLE:
            if smoothed_wrist_y < shoulder_y:
                self.state = STATE_RISING
                self._buffer = [self._frame_metrics(landmarks, side)]
        else:
            self._buffer.append(self._frame_metrics(landmarks, side))
            if self.state == STATE_RISING and smoothed_wrist_y < release_y:
                self.state = STATE_PEAKED
            elif self.state == STATE_PEAKED and smoothed_wrist_y > shoulder_y:
                if len(self._buffer) >= self.min_rep_frames:
                    self._complete_rep()
                else:
                    # Too short to be a real shot cycle -- likely jitter
                    # crossing the threshold rather than an actual rep.
                    self._buffer = []
                    self.state = STATE_IDLE

        return self.state

    @staticmethod
    def _frame_metrics(landmarks, side):
        return {
            "elbow_angle": angle_math.elbow_angle(landmarks, side),
            "knee_angle": angle_math.knee_angle(landmarks, side),
            "elbow_lateral_deviation": angle_math.elbow_lateral_deviation(landmarks, side),
            "base_width_ratio": angle_math.base_width_ratio(landmarks),
            "wrist_separation": angle_math.wrist_separation(landmarks, side),
        }

    def manual_count_rep(self):
        """Force-complete the current buffer as a rep, for a shot the
        auto-detector missed."""
        if self._buffer:
            return self._complete_rep()
        return None

    def discard_current_rep(self):
        """Clear the current buffer without counting it, for motion the
        auto-detector mistook for a shot."""
        self._buffer = []
        self.state = STATE_IDLE

    def _complete_rep(self):
        self.rep_count += 1
        rep_record = {"rep_number": self.rep_count, "frames": self._buffer}
        self._buffer = []
        self.state = STATE_IDLE
        self._cooldown = COOLDOWN_FRAMES
        if self.on_rep_complete:
            self.on_rep_complete(rep_record)
        return rep_record
