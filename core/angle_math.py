"""Per-frame geometry: joint angles and positional checks used by the
problem-bank metrics in data/problems.json (flying elbow, knee bend, base
width, etc.). Operates on a single frame's landmark list, as returned by
PoseLandmarker.detect_for_video(...).pose_landmarks[0].

Time-series metrics (variance across reps, guide-hand release timing) build
on top of these per-frame functions and live in rep_detection.py instead.
"""

import math

# BlazePose landmark indices (33-point topology), named for the joints
# angle_math.py actually needs.
LANDMARK = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}


def _point(landmarks, name):
    lm = landmarks[LANDMARK[name]]
    return (lm.x, lm.y)


def calculate_angle(a, b, c):
    """Angle at vertex b, in degrees, formed by points a-b-c."""
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    ang = abs(ang)
    return 360 - ang if ang > 180 else ang


def elbow_angle(landmarks, side):
    """Shoulder-elbow-wrist angle. ~180 deg = fully extended (release),
    smaller = bent (load)."""
    shoulder = _point(landmarks, f"{side}_shoulder")
    elbow = _point(landmarks, f"{side}_elbow")
    wrist = _point(landmarks, f"{side}_wrist")
    return calculate_angle(shoulder, elbow, wrist)


def knee_angle(landmarks, side):
    """Hip-knee-ankle angle. ~180 deg = straight leg, smaller = bent (load)."""
    hip = _point(landmarks, f"{side}_hip")
    knee = _point(landmarks, f"{side}_knee")
    ankle = _point(landmarks, f"{side}_ankle")
    return calculate_angle(hip, knee, ankle)


def elbow_lateral_deviation(landmarks, side):
    """Signed horizontal distance from the elbow to the shoulder-hip
    vertical line, normalized by torso width (shoulder-to-shoulder distance)
    so it's comparable across different distances from the camera.

    Used for the "flying elbow" check (front/back view): near 0 means the
    elbow is tucked under the body line, larger means it's flared outward.
    """
    shoulder = _point(landmarks, f"{side}_shoulder")
    hip = _point(landmarks, f"{side}_hip")
    elbow = _point(landmarks, f"{side}_elbow")
    torso_line_x = (shoulder[0] + hip[0]) / 2

    other_side = "right" if side == "left" else "left"
    torso_width = abs(
        _point(landmarks, f"{side}_shoulder")[0]
        - _point(landmarks, f"{other_side}_shoulder")[0]
    )
    if torso_width == 0:
        return 0.0

    return (elbow[0] - torso_line_x) / torso_width


def base_width_ratio(landmarks):
    """Ankle-to-ankle distance divided by shoulder-to-shoulder distance.
    Around 1.0 is roughly shoulder-width stance; well below or above that
    flags a base that's too narrow or too wide.
    """
    left_ankle = _point(landmarks, "left_ankle")
    right_ankle = _point(landmarks, "right_ankle")
    left_shoulder = _point(landmarks, "left_shoulder")
    right_shoulder = _point(landmarks, "right_shoulder")

    shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
    if shoulder_width == 0:
        return 0.0

    ankle_width = abs(left_ankle[0] - right_ankle[0])
    return ankle_width / shoulder_width


def wrist_separation(landmarks, shooting_side):
    """Distance between the shooting wrist and the guide-hand wrist,
    normalized by torso width. Used to help detect guide-hand timing
    (e.g. the guide hand separating from the ball) on the front view.
    """
    guide_side = "right" if shooting_side == "left" else "left"
    shooting_wrist = _point(landmarks, f"{shooting_side}_wrist")
    guide_wrist = _point(landmarks, f"{guide_side}_wrist")

    left_shoulder = _point(landmarks, "left_shoulder")
    right_shoulder = _point(landmarks, "right_shoulder")
    torso_width = abs(left_shoulder[0] - right_shoulder[0])
    if torso_width == 0:
        return 0.0

    dx = shooting_wrist[0] - guide_wrist[0]
    dy = shooting_wrist[1] - guide_wrist[1]
    return math.hypot(dx, dy) / torso_width
