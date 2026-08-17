# Basketball Shooting Form Coach

A webcam-based app that analyzes basketball shooting form using pose estimation and gives structured coaching feedback and corrective drills.

## What it does

1. **Fundamentals screen** — before any analysis, shows 9 baseline shooting cues (stance, hand placement, elbow position, follow-through, etc.), each with a short self-recorded demo clip.
2. **Assessment session** — the user shoots 5 reps from each of 4 camera angles (front, back, shooter's-strong-side, shooter's-guide-side, determined by a handedness setting chosen at the start). Reps are auto-detected via a motion state machine, with a manual override if a rep is miscounted.
3. **Analysis** — MediaPipe pose landmarks are used to compute joint angles and positions per rep, aggregated across reps per angle to separate real form issues from one-off noise.
4. **Feedback** — detected deviations are matched against a data-driven problem bank (`data/problems.json`), each mapped to a likely cause and a specific corrective drill. An LLM (Claude API) turns the raw metrics into natural-language coaching tips.

## Tech stack

- **Python** — core language
- **MediaPipe** — pose estimation (33 body landmarks)
- **OpenCV** — frame capture and skeleton overlay drawing
- **Streamlit** + **streamlit-webrtc** — web UI and real-time webcam handling
- **Claude API** — translates measured metrics into readable coaching feedback

## Project structure

```
basketball-form-coach/
├── docs/
│   └── design-log.md      # decision log: what was decided, why, and how AI was used at each step
├── data/
│   ├── fundamentals.json  # the 9 baseline coaching cues
│   └── problems.json      # problem -> cause -> drill data bank
├── app/                    # Streamlit UI (in progress)
├── core/                   # pose estimation, angle math, rep detection (in progress)
├── assets/                 # self-recorded fundamentals clips (in progress)
└── requirements.txt
```

## Status

In development. See [docs/design-log.md](docs/design-log.md) for the full design history and reasoning behind each decision.
