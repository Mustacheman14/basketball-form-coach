"""Shot-cycle rep detection: segments a continuous stream of pose frames
into individual shot reps and counts them automatically.

Cycle signal is the shooting wrist's height relative to the shoulder and
nose (MediaPipe's y increases downward, so "above" means a smaller y value):

    IDLE (wrist below shoulder)
      -> RISING (wrist crosses above shoulder)
      -> PEAKED (wrist crosses above the nose, i.e. a release-height event)
      -> back to IDLE (wrist drops below shoulder again) => rep counted

Each completed rep returns the buffered per-frame metrics (from
angle_math.py) collected during that rep, for later aggregation across the
5 reps in a session. Auto-detection can misfire (motion outside a real shot,
or a missed cycle), so manual_count_rep() and discard_current_rep() exist
as the override a user can trigger from the UI.
"""

from core import angle_math

STATE_IDLE = "idle"
STATE_RISING = "rising"
STATE_PEAKED = "peaked"


class RepCounter:
    def __init__(self, shooting_side, on_rep_complete=None):
        self.shooting_side = shooting_side
        self.on_rep_complete = on_rep_complete
        self.state = STATE_IDLE
        self.rep_count = 0
        self._buffer = []

    def reset(self):
        self.state = STATE_IDLE
        self.rep_count = 0
        self._buffer = []

    def update(self, landmarks):
        """Feed one frame's landmarks in. Returns the current state."""
        side = self.shooting_side
        wrist_y = landmarks[angle_math.LANDMARK[f"{side}_wrist"]].y
        shoulder_y = landmarks[angle_math.LANDMARK[f"{side}_shoulder"]].y
        nose_y = landmarks[0].y

        frame_metrics = {
            "elbow_angle": angle_math.elbow_angle(landmarks, side),
            "knee_angle": angle_math.knee_angle(landmarks, side),
            "elbow_lateral_deviation": angle_math.elbow_lateral_deviation(landmarks, side),
            "base_width_ratio": angle_math.base_width_ratio(landmarks),
            "wrist_separation": angle_math.wrist_separation(landmarks, side),
        }

        if self.state == STATE_IDLE:
            if wrist_y < shoulder_y:
                self.state = STATE_RISING
                self._buffer = [frame_metrics]
        else:
            self._buffer.append(frame_metrics)
            if self.state == STATE_RISING and wrist_y < nose_y:
                self.state = STATE_PEAKED
            elif self.state == STATE_PEAKED and wrist_y > shoulder_y:
                self._complete_rep()

        return self.state

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
        if self.on_rep_complete:
            self.on_rep_complete(rep_record)
        return rep_record
