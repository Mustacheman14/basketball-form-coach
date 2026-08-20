# Design Log

This log tracks the major design decisions behind this project, why each was made, and how AI (Claude) was used at each step — as research assistant, technical sounding board, and implementation partner. Entries are in the order decisions were made, not necessarily the order features ship.

---

## 1. Concept and scope

**Decision:** Narrow from "basketball form analysis" broadly down to a single skill: jump-shot shooting form. Specifically the mechanics most beginners get wrong — elbow position, guide-hand behavior, base, follow-through, timing.

**Why:** A single, well-executed skill analyzer is more finishable and more credible than a shallow multi-skill tool. It also keeps the biomechanics claims narrow enough to be defensible with cited sources rather than guessed thresholds.

**AI's role:** Claude pushed back on the original broad scope and argued for narrowing to one movement before any code was written, then helped separate "what MediaPipe can realistically track" (body joints) from "what it can't" (individual fingers), which shaped every downstream decision.

---

## 2. Camera angles and heights

**Decision:** Each form checkpoint is measured from whichever angle actually captures it cleanly:

| Angle | Captures | Height |
|---|---|---|
| Front | Eyes/head direction, guide-hand behavior, shoulder squareness | Chest/shoulder height |
| Side | Elbow flexion/extension, follow-through, arc, one-motion vs. two-motion timing, knee bend | Chest/shoulder height, camera perpendicular to the motion plane |
| Back | Elbow tuck/alignment (less occluded than front), shoulder-hip symmetry, landing drift | Chest/shoulder height |

**Why:** No single angle captures every checkpoint — a front-only setup, the original plan, would have completely missed elbow tuck and knee-bend mechanics. Camera height was set near chest/shoulder level specifically to minimize perspective foreshortening in the joint-angle math, since MediaPipe's 2D angle calculations get less accurate the more a camera looks up or down at a joint.

**AI's role:** Claude researched real coaching video-analysis guidance (cited below) to identify what each angle is actually good for, rather than guessing, and flagged that back view — not front view, the original assumption — is the stronger single angle for elbow alignment specifically, because the elbow's silhouette is less obstructed by the head/ball from behind.

**Sources:**
- [FORM SHOOTING - BEEF SET-UP](https://www.ballaratbasketball.com/wp-content/uploads/2024/06/SHOOTING.pdf)
- [Basketball Shot Analysis: Break Down Your Shot Like a Pro | Coach Dave](https://coachdavelove.com/all-about-video-shot-analysis-how-to-break-down-a-shot-like-a-pro/)
- [How to Film Your Athlete for Video Analysis Lessons](https://coach.ly/blog/how-to-film-your-athlete-for-video-analysis-lessons)

---

## 3. Handedness

**Decision:** Ask the user explicitly ("right-handed or left-handed?") at the start of a session rather than trying to auto-detect it from footage. The answer determines which side's landmarks (`LEFT_*` vs `RIGHT_*`) feed the elbow/guide-hand metrics, and relabels the physical left/right camera positions as "shooting-arm side" vs "guide-hand side."

**Why:** Every downstream metric depends on picking the correct arm. Auto-detection (e.g. inferring dominant hand from which hand is under the ball) is an interesting idea but fragile — a bad guess would silently corrupt an entire session's data. Reliability was prioritized over cleverness for a value everything else depends on.

**AI's role:** Claude framed this as an explicit reliability-vs-cleverness tradeoff and recommended the explicit-question approach, logging auto-detection as a documented future-work idea rather than a v1 dependency.

---

## 4. Session flow

**Decision:** A recording session collects 5 reps at each of the 4 angles (front, back, strong-side, guide-side), in sequence. Reps are counted automatically via a state machine over shooting-elbow angle / wrist height (`rest -> loading -> release -> follow-through -> rest` = 1 rep). Once 5 reps are counted, the app prompts the user to move to the next angle. A manual override lets the user correct a miscount.

**Why:** A single webcam can't capture all 4 angles simultaneously, so this necessarily became a sequential, multi-angle assessment session rather than a single continuous live feed — a deliberate departure from the original "always live" pitch. Recording multiple reps per angle (rather than judging on one shot) lets the app read consistency, not just average form — e.g. an elbow angle that varies a lot rep to rep is itself useful feedback, independent of whether the average is "correct."

**AI's role:** Claude flagged the single-webcam constraint explicitly (it's physically impossible to get 4 angles from 1 continuous feed) and proposed splitting the product into two modes: a deeper multi-angle **assessment mode** and a lighter single-angle **live mode** for demo purposes. It also proposed the auto-detect state machine and flagged its main risk (silent miscounts corrupting aggregated data) before it was built, which is why the manual override exists.

---

## 5. Problem -> drill data bank

**Decision:** Detected form issues are matched against a structured data file (`data/problems.json`) that maps each problem to: which camera angle detects it, the specific pose metric used, its likely cause, and a corrective drill. See that file for the current set (flying elbow, thumbing the ball, two-motion shot, inconsistent release, base/balance issues, guide-hand timing, knee bend/leg drive).

**Why:** Keeping this as data rather than hardcoded logic means the coaching content has one canonical, editable source of truth, and it's independently checkable against sources rather than buried in code.

**AI's role:** Claude researched real coaching material on common shooting mistakes and their corrective drills, and structured the findings into the schema used in `problems.json`.

**Sources:**
- [A 3 Part Plan To Address The 2 Hand Shot Problem](https://www.breakthroughbasketball.com/training/fixing-thumb-flick)
- [Basketball Shooting Drill: Elbow Shooting](https://www.breakthroughbasketball.com/drills/elbowshooting)
- [12 Drills to Fix Common Shooting Mistakes and Build Better Form](https://www.breakthroughbasketball.com/drills/shot-doctor-fixes)
- [Basketball Shooting Fundamentals, Form and Technique](https://www.coachesclipboard.net/Shooting.html)

---

## 6. Fundamentals screen

**Decision:** Before any analysis, every user sees 9 baseline shooting cues (`data/fundamentals.json`), each paired with a short self-recorded video clip shot in the same camera angles used elsewhere in the app.

**Why:** General fundamentals (stance, hand placement, facing the basket, release timing, etc.) apply to everyone regardless of what the pose analysis later detects, so they're handled as static onboarding content rather than reactive feedback. Clips are self-recorded rather than sourced from professional game footage, since embedding copyrighted broadcast footage (even with credit) in a public repo would be redistribution, not citation — self-recorded clips also double as authentic demo material.

**AI's role:** Claude researched jump-shot release timing specifically, since the initial assumption ("release at the top of the jump") turned out to be imprecise — sources favor releasing on the way up, since jump velocity (and therefore leg power transferred to the shot) is highest right after leaving the ground and drops to zero at the apex. Claude also flagged the copyright issue with using pro footage and proposed the self-recorded approach.

**Sources:**
- [Biomechanics of the Basketball Jump Shot - Physiopedia](https://www.physio-pedia.com/Biomechanics_of_the_Basketball_Jump_Shot)
- [Basketball: Release the Ball Early for Better Jump Shots](https://www.physicaleducationupdate.com/public/329.cfm)

---

## 7. Tech stack

**Decision:** Python, MediaPipe, OpenCV, Streamlit + `streamlit-webrtc`, Claude API.

**Why:** All free/open-source except the Claude API, which is used narrowly — to translate already-computed metrics into readable coaching language, not to do the biomechanical analysis itself. Plain Streamlit has no real webcam loop primitive; `streamlit-webrtc` was added specifically to avoid the laggy `st.image()`-in-a-rerun-loop workaround.

**AI's role:** Claude flagged the `streamlit-webrtc` gap before it became a problem discovered mid-build, and helped scope what the Claude API should and shouldn't be responsible for (language generation, not measurement) so the feedback stays grounded in actual pose data rather than the LLM inventing plausible-sounding but unverified critique.

---

## 8. First working prototype: pose estimation smoke test

**Decision:** `core/pose_estimation.py` opens the webcam, runs MediaPipe pose detection on each frame, and draws a skeleton overlay using OpenCV.

**What went wrong first:** The `requirements.txt`-installed `mediapipe` version (1.0.1) turned out to have completely removed the legacy `mp.solutions.pose` API that most tutorials and the original draft of this script were written against — `import mediapipe as mp; mp.solutions` raised `AttributeError`, and even the direct submodule path `mediapipe.python.solutions` no longer exists. This wasn't caught until actually running the code against a live webcam frame.

**Fix:** Rewrote against MediaPipe's current Tasks API (`mediapipe.tasks.python.vision.PoseLandmarker`), which requires downloading a separate `.task` model file rather than using a bundled model. Landmark drawing is now done manually with OpenCV (`cv2.line`/`cv2.circle`) using the standard 33-point BlazePose body connection topology, since the old `drawing_utils` helper went away along with `mp.solutions`.

**Why this is worth logging:** This is a good example of a plan meeting reality — the library had moved on since most of the tutorials/documentation online were written, and the fix was found by actually running the code against a real webcam frame rather than assuming the first draft worked.

**AI's role:** Claude wrote the initial script, hit the `AttributeError` when testing it, researched MediaPipe's current official documentation to confirm the Tasks API is now the only supported path, downloaded the correct model asset from Google's official model hosting, and rewrote the script around it — verified end-to-end against a live webcam frame (33 landmarks successfully detected) before considering the piece done.

**Sources:**
- [Pose landmark detection guide for Python | Google AI Edge](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)

---

## 9. Joint-angle geometry module

**Decision:** `core/angle_math.py` computes the per-frame geometric values the problem bank needs: elbow angle (shoulder-elbow-wrist), knee angle (hip-knee-ankle), elbow lateral deviation (flying-elbow check), base width ratio (ankle spread vs. shoulder width), and wrist separation (guide-hand proximity check). Each takes a `side` ("left"/"right") argument rather than assuming a shooting hand, so it plugs directly into the handedness setting from decision #3. Values are normalized against torso/shoulder width where relevant, so they stay comparable regardless of how far the shooter is from the camera.

**Why:** These are per-frame primitives only — variance across reps and event timing (e.g. when the guide hand separates from the ball) are time-series concerns that belong in the rep-detection state machine (decision #4), not here. Keeping this module to single-frame geometry keeps it independently testable.

**AI's role:** Claude wrote the module and verified it against a live webcam frame before committing (angles came back in a plausible 0-180 degree range, base width ratio and lateral deviation in sane proportions) rather than trusting the math would work untested.

---

## 10. Rep detection state machine

**Decision:** `core/rep_detection.py` implements the auto-detect rep counter from decision #4 as a 3-state machine: `idle -> rising` (shooting wrist crosses above the shoulder) `-> peaked` (wrist crosses above the nose, i.e. a release-height event) `-> back to idle` (wrist drops below the shoulder again), at which point one rep is counted. Per-frame metrics from `angle_math.py` are buffered during the active states and handed back as one record per completed rep, for aggregation across the 5 reps later. `manual_count_rep()` and `discard_current_rep()` implement the manual-override safety valve committed to in decision #4.

**Why wrist-height-relative-to-shoulder/nose, not just elbow angle alone:** a shooter's elbow is bent both when holding the ball at rest and when loading a shot, so elbow angle alone doesn't cleanly separate "idle" from "about to shoot." Wrist height gives a clean rise-and-fall signal that actually matches the physical shape of a shot.

**Limitation, honestly logged:** this can misfire on motion that isn't a real shot (e.g. raising a hand for another reason), which is exactly why the manual override exists rather than trusting auto-detection blindly. Real validation requires live testing against actual shooting motion — a scripted smoke test can only confirm the code runs without errors, not that it counts real shots correctly.

**AI's role:** Claude designed the state machine, chose the wrist-height signal over elbow-angle-only after reasoning through why the latter is ambiguous at rest, implemented the manual-override methods per the earlier reliability decision, and ran a live 60-frame smoke test to confirm no exceptions before commit — full behavioral validation (does it count a real shot correctly) is flagged as needing the user's own live testing.

**Follow-up bug found via real testing:** live testing (by the user, not an automated test) surfaced occasional double-counting. Root cause: the original state machine counted a rep as soon as the wrist crossed back below the shoulder line once, with no minimum duration, so landmark-tracking jitter during the arm's descent after a shot could flicker across that threshold for a couple of frames and register as a second, near-instant phantom rep. Fixed by requiring a completed rep to span at least `MIN_REP_FRAMES` (10) frames before it's accepted; shorter cycles are treated as noise and discarded rather than counted. `manual_count_rep()` intentionally bypasses this minimum, since that's an explicit user confirmation, not an inference.

**Why this is worth logging:** the bug only showed up under real usage, not the earlier synthetic smoke test — a good example of why the design log distinguishes "code runs without errors" from "code behaves correctly," and why live testing by an actual user was called out as necessary rather than skipped.
