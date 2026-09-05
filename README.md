# Sentry

A webcam sentry that recognises a face, makes the person prove their hands are
their own, and then reads gestures from those hands - refusing anything from a
hand it cannot account for.

## The idea

A face can be identified. A hand cannot: there is no biometric signature in a
wrist. So the identity lives on the face and stays there, and hands are only
ever checked for **whether they belong to it**.

That check is the chain. The pose model reports keypoints per person, so a wrist
is trusted only when its whole shoulder-elbow-wrist chain is visible on the
skeleton the recognised face sits on. A hand reaching in from somewhere else has
no chain leading back to that face, and nothing it does will be obeyed.

```
face  ->  who this is          (insightface, buffalo_l)
pose  ->  whose arm this is    (YOLO11-pose)
hand  ->  what the fingers do  (MediaPipe HandLandmarker)
```

## Flow

| State | Meaning | Leaves when |
|---|---|---|
| `IDLE` | nobody recognised | a known face holds for 3 of 5 frames |
| `AWAITING_HANDS` | "show both hands" | both chained wrists are visible |
| `BOUND` | verified, gestures live | the face leaves the frame |
| `WARN` | face gone, counting down | it comes back, or the 5s lease expires |

Once `BOUND` the hands are free to move out of shot. The identity is anchored to
the face, so only losing the face breaks it.

**Alarms:** `UNKNOWN PERSON` (a face that matches nobody), `UNCHAINED HAND` (a
wrist that belongs to no verified arm), `NO BODY CHAIN` (face recognised but no
skeleton to attach it to - usually standing too close).

Every one of these is majority-voted over 5 frames. A single bad frame never
changes anything.

## Setup

Python 3.11. Everything installs into a local venv:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Then remove the CPU build of onnxruntime, which insightface pulls in and which
shadows the GPU one:

```bash
.venv/Scripts/python.exe -m pip uninstall -y onnxruntime
.venv/Scripts/python.exe -m pip install --force-reinstall --no-deps onnxruntime-gpu
```

For CUDA on Windows, install the CUDA build of torch:

```bash
.venv/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.14.0+cu130" "torchvision==0.29.0+cu130"
```

NVIDIA publishes the `nvidia-*-cu13` runtime wheels for Linux only, so there is
no pip route to the CUDA DLLs on Windows. The CUDA build of torch bundles the
same DLLs in `torch/lib`, and `gpu.py` points onnxruntime at them - no
system-wide CUDA toolkit needed. Without it everything still runs, on the CPU.

Models download themselves into `models/` on first run (~350 MB).

## Usage

Note the interpreter: `py` and `python` may point at a different install.

```bash
.venv/Scripts/python.exe enroll.py     # SPACE to capture, Q to finish
.venv/Scripts/python.exe main.py       # Q or ESC to quit
```

Enrol from several angles - the recogniser keeps every vector and matches
against the closest one, so variety helps.

## Permissions

`permissions.toml` decides who may do what. Gestures are only read from a
`BOUND` identity, so nothing written here can be triggered by a stranger.

```toml
[permissions]
Mert = ["*"]                        # every gesture
"*" = ["hands_up", "thumbs_up"]     # anyone recognised
```

Unlisted person, unlisted gesture and unidentified caller are all denied.
Unknown gesture names are reported at startup, because a typo would otherwise
fail silently as a refusal.

| Arm poses | Finger poses |
|---|---|
| `hands_up`, `arms_crossed`, `t_pose` | `thumbs_up`, `thumbs_down`, `peace` |
| `left_arm_raised`, `right_arm_raised` | `pointing`, `open_palm`, `fist` |

Left and right are the person's own, not the side of the screen. Hook a real
action into `on_gesture` in `main.py`; it prints by default.

## Files

| | |
|---|---|
| `main.py` | the loop |
| `enroll.py` | face enrolment |
| `gpu.py` | CUDA bootstrap, import first |
| `face_detector.py` `recognizer.py` | identity, plus the frame-voting `Voter` |
| `pose_detector.py` | skeletons and the arm chain |
| `hand_detector.py` `hand_gestures.py` | landmarks and finger shapes |
| `binder.py` | chain checks and the state machine |
| `gestures.py` `permissions.py` | arm poses, gesture latch, access rules |

## Tuning

| Where | Default | |
|---|---|---|
| `recognizer.py` | `threshold=0.38` | face match; lower accepts more |
| `binder.py` | `lease=5.0` | seconds the identity survives a missing face |
| `binder.py` | `hands_timeout=15.0` | seconds to show both hands |
| `binder.py` | `MAX_REACH=6.0` | wrist distance from the head, in face widths |
| `binder.py` | `HAND_MATCH=1.5` | hand-to-wrist match distance, same unit |
| `main.py` | `FACE_EVERY=3` | run the face model every Nth frame while `BOUND` |
| `recognizer.py` | `Voter(5, 3)` | frames voted, votes needed |

Distances are measured in face widths rather than pixels, so they hold at any
distance from the camera.

## Performance

Roughly 20-25 fps on a GTX 1660 Ti at 640x480, with the camera itself a large
part of the remaining cost. The hand model runs on the CPU in a worker thread
alongside the GPU models, and the face model is thinned to every third frame
once the identity is settled - neither costs any accuracy.

## Limits

- One face at a time: the largest face in frame is the one identified. The pose
  model still sees everyone else, which is what the intruder checks rely on.
- Handedness from MediaPipe assumes a mirrored image and is not used. Hands are
  tied to people by wrist position instead.
- A hand is only read for fingers once it matches a chained wrist, so a person
  standing at an angle that hides a shoulder loses that arm until it reappears.
