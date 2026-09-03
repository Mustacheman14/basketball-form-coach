"""Multi-angle assessment session: sequences the 4 camera angles (front,
strong-side, guide-side, back) from design-log.md decision #4, running a
RepCounter for 5 reps at each angle before prompting the user to move to
the next one.

"strong-side" / "guide-side" are named relative to the shooting hand rather
than literal room-left/room-right, so the same session works for both
right- and left-handed shooters without swapping any logic -- only the
prompt text differs.
"""

from core.rep_detection import RepCounter

REPS_PER_ANGLE = 5

ANGLE_SEQUENCE = ["front", "strong_side", "guide_side", "back"]

# Kept as short separate lines rather than one long string -- a single
# concatenated sentence was running past the edge of the video frame.
CAMERA_HEIGHT_LINE = "Camera at chest/shoulder height (not waist, not overhead)."

ANGLE_PROMPTS = {
    "front": [
        "Face the camera directly.",
        CAMERA_HEIGHT_LINE,
    ],
    "strong_side": [
        "Turn so your shooting arm faces the camera (stand sideways).",
        CAMERA_HEIGHT_LINE,
    ],
    "guide_side": [
        "Turn so your guide-hand arm faces the camera (other side).",
        CAMERA_HEIGHT_LINE,
    ],
    "back": [
        "Turn so your back faces the camera.",
        CAMERA_HEIGHT_LINE,
    ],
}

SESSION_COMPLETE_PROMPT = ["Session complete."]

# Shown once before the session starts, not per angle -- distance from the
# camera and from the hoop doesn't change between angles.
PRE_SESSION_TIPS = [
    "Stand about 8-10 ft back from the camera so your whole body fits in",
    "frame, with some space above your head and around your feet.",
    "",
    "Shoot from a spot close enough that you make most of your shots",
    "(e.g. the free-throw line or closer) -- this session is about form,",
    "not range.",
]


class AssessmentSession:
    def __init__(self, shooting_side, reps_per_angle=REPS_PER_ANGLE, on_rep_complete=None):
        self.shooting_side = shooting_side
        self.reps_per_angle = reps_per_angle
        self.angle_index = 0
        self.results = {angle: [] for angle in ANGLE_SEQUENCE}
        self._external_on_rep_complete = on_rep_complete
        self.counter = RepCounter(
            shooting_side=shooting_side, on_rep_complete=self._on_rep_complete
        )

    @property
    def current_angle(self):
        if self.is_complete:
            return None
        return ANGLE_SEQUENCE[self.angle_index]

    @property
    def current_prompt_lines(self):
        angle = self.current_angle
        return ANGLE_PROMPTS[angle] if angle else SESSION_COMPLETE_PROMPT

    @property
    def reps_done_this_angle(self):
        angle = self.current_angle
        return len(self.results[angle]) if angle else 0

    @property
    def is_complete(self):
        return self.angle_index >= len(ANGLE_SEQUENCE)

    def update(self, landmarks):
        if self.is_complete:
            return "complete"
        return self.counter.update(landmarks)

    def manual_count_rep(self):
        if self.is_complete:
            return None
        return self.counter.manual_count_rep()

    def discard_current_rep(self):
        if not self.is_complete:
            self.counter.discard_current_rep()

    def skip_to_next_angle(self):
        """Escape hatch: move on even if fewer than reps_per_angle were
        captured, for a session that isn't converging cleanly."""
        if not self.is_complete:
            self._advance_angle()

    def remove_last_rep(self):
        """Undo the most recently recorded rep. Handles the case where the
        phantom rep was the one that just triggered auto-advance to a new
        (otherwise empty) angle, by stepping back to the previous angle."""
        angle = self.current_angle
        if angle is None or not self.results[angle]:
            if self.angle_index == 0:
                return None
            angle = ANGLE_SEQUENCE[self.angle_index - 1]
            if not self.results[angle]:
                return None
            removed = self.results[angle].pop()
            self.angle_index -= 1
            self.counter.reset()
            return removed

        removed = self.results[angle].pop()
        self.counter.rep_count = len(self.results[angle])
        return removed

    def _on_rep_complete(self, rep_record):
        angle = self.current_angle
        if angle is None:
            return
        rep_record["angle"] = angle
        self.results[angle].append(rep_record)
        if self._external_on_rep_complete:
            self._external_on_rep_complete(rep_record)
        if len(self.results[angle]) >= self.reps_per_angle:
            self._advance_angle()

    def _advance_angle(self):
        self.angle_index += 1
        self.counter.reset()
