"""First-pass feedback engine: turns a completed AssessmentSession's raw
per-rep metrics and shot outcomes into a shooting percentage/miss breakdown
(fully data-driven, no calibration needed) and a set of possible form
issues drawn from data/problems.json (heuristic thresholds -- see the
THRESHOLDS caveat below).
"""

import json
import statistics

PROBLEMS_PATH = "data/problems.json"

# These thresholds are first-pass estimates based on the geometry each
# metric measures (see angle_math.py's docstrings for what each value
# means), not on calibration against real coaching footage or a labeled
# dataset -- we deliberately dropped using pro game footage for copyright
# reasons (see design-log.md decision on the fundamentals screen), so there
# is no ground truth to tune against yet. Treat flags from this module as
# "worth a second look," not a diagnosis. Revisit these once there's
# outcome data (make%) to correlate against.
THRESHOLDS = {
    "flying_elbow": 0.30,       # abs(elbow_lateral_deviation) above this flags flaring
    "inconsistent_release": 15.0,  # stdev of release-frame elbow angle, in degrees
    "base_balance_low": 0.7,    # base_width_ratio below this = stance too narrow
    "base_balance_high": 1.4,   # base_width_ratio above this = stance too wide
    "knee_bend": 165.0,         # knee angle at load above this = barely bending knees
}


def load_problems():
    with open(PROBLEMS_PATH) as f:
        return {p["id"]: p for p in json.load(f)["problems"]}


def _release_frame(frames):
    """The frame with the highest elbow angle approximates the release
    instant (arm closest to fully extended)."""
    if not frames:
        return None
    return max(frames, key=lambda f: f["elbow_angle"])


def _rep_has_hitch(frames):
    """Simple two-motion/hitch proxy: does elbow angle dip after already
    having risen partway toward its peak, instead of rising smoothly?
    A real hitch shows up as a temporary drop mid-rise rather than a
    monotonic climb to the release frame."""
    angles = [f["elbow_angle"] for f in frames]
    if len(angles) < 5:
        return False
    peak = max(angles)
    peak_idx = angles.index(peak)
    rising_segment = angles[: peak_idx + 1]
    # Allow small noise dips, but a drop of more than ~8 degrees during the
    # rise is treated as a real hitch rather than tracking jitter.
    for i in range(1, len(rising_segment)):
        if rising_segment[i] < rising_segment[i - 1] - 8:
            return True
    return False


def shooting_stats(results):
    """Make/miss breakdown across every rep with a logged outcome, ignoring
    reps where the outcome question was skipped."""
    makes = 0
    misses = 0
    miss_type_counts = {}

    for reps in results.values():
        for rep in reps:
            outcome = rep.get("outcome")
            if not outcome:
                continue
            if outcome.get("made"):
                makes += 1
            else:
                misses += 1
                miss_type = outcome.get("miss_type")
                if miss_type:
                    miss_type_counts[miss_type] = miss_type_counts.get(miss_type, 0) + 1

    total = makes + misses
    return {
        "makes": makes,
        "misses": misses,
        "total": total,
        "make_pct": (makes / total * 100) if total else None,
        "miss_type_counts": miss_type_counts,
    }


def analyze_form(results):
    """Returns a list of {problem, cause, drill} dicts for form issues the
    aggregated metrics suggest are worth a look. See THRESHOLDS caveat."""
    problems = load_problems()
    flags = []

    def angle_frames(angle):
        return [rep["frames"] for rep in results.get(angle, []) if rep["frames"]]

    # Every check below evaluates each relevant angle SEPARATELY and flags
    # if ANY one of them crosses the threshold, rather than pooling all
    # angles into one blended average. Pooling was tried first and found to
    # let a clean-looking angle mask a flagrant one from another angle
    # (verified with synthetic data before this was caught) -- a false
    # negative is worse than an extra "worth a look" flag here, since flags
    # are already hedged as suggestions, not diagnoses.

    # Flying elbow: front and back independently, elbow lateral deviation
    # at release.
    for angle in problems["flying_elbow"]["detected_from"]:
        deviations = [
            _release_frame(frames)["elbow_lateral_deviation"]
            for frames in angle_frames(angle)
            if _release_frame(frames)
        ]
        if deviations and (sum(abs(d) for d in deviations) / len(deviations)) > THRESHOLDS["flying_elbow"]:
            flags.append(problems["flying_elbow"])
            break

    # Two-motion / hitch: strong_side and guide_side, majority-of-reps check
    # (naturally combined since a hitch is a per-rep yes/no, not an average).
    side_reps = []
    for angle in problems["two_motion"]["detected_from"]:
        side_reps.extend(angle_frames(angle))
    if side_reps:
        hitch_count = sum(1 for frames in side_reps if _rep_has_hitch(frames))
        if hitch_count > len(side_reps) / 2:
            flags.append(problems["two_motion"])

    # Inconsistent release: stdev of release-frame elbow angle, checked
    # independently per side angle.
    for angle in problems["inconsistent_release"]["detected_from"]:
        release_angles = [
            _release_frame(frames)["elbow_angle"]
            for frames in angle_frames(angle)
            if _release_frame(frames)
        ]
        if len(release_angles) >= 3 and statistics.stdev(release_angles) > THRESHOLDS["inconsistent_release"]:
            flags.append(problems["inconsistent_release"])
            break

    # Base/balance: front and back independently (side angles excluded --
    # see problems.json note on why that measurement isn't valid there).
    for angle in problems["base_balance"]["detected_from"]:
        base_widths = [frames[0]["base_width_ratio"] for frames in angle_frames(angle)]
        if base_widths:
            avg_width = sum(base_widths) / len(base_widths)
            if avg_width < THRESHOLDS["base_balance_low"] or avg_width > THRESHOLDS["base_balance_high"]:
                flags.append(problems["base_balance"])
                break

    # Knee bend: side angles independently, minimum knee angle per rep
    # (deepest load) averaged within each angle.
    for angle in problems["knee_bend"]["detected_from"]:
        knee_mins = [min(f["knee_angle"] for f in frames) for frames in angle_frames(angle)]
        if knee_mins and (sum(knee_mins) / len(knee_mins)) > THRESHOLDS["knee_bend"]:
            flags.append(problems["knee_bend"])
            break

    return flags


def generate_report(results):
    """Top-level entry point: combines shooting stats and form flags into
    one report dict for display."""
    return {
        "stats": shooting_stats(results),
        "form_flags": analyze_form(results),
    }
