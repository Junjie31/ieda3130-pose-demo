# IEDA 3130 · Pose Estimation Demo (OpenPose-style, built on MediaPipe)

> HKUST · Department of Industrial Engineering and Decision Analytics  
> IEDA 3130 Ergonomics and Safety Management · Fall 2026  
> Instructor: Prof. Richard SO · TA: Junjie HOU

This demo shows how a camera + a pose-estimation network can extract the body
landmarks that we measured **by hand with an anthropometer in Lab 2**
(Anthropometry: Workstation Design), and how to judge a seated posture against
the **BS 7310** criterion (elbow ≈ 90°, forearm horizontal at keyboard height).

It is intentionally small: **4 short Python files, 3 pip packages, no GPU, no
compilation** (the 4th file is an optional unit test). Everything runs on a
normal laptop (Windows / macOS / Linux) or in Google Colab.

| | OpenPose (CMU, 2017) | This demo (MediaPipe Pose Landmarker) |
|---|---|---|
| Keypoints | 18 / 25 body (+hands/face) | 33 body landmarks, 2-D + 3-D "world" coords |
| Install | C++/CUDA/Caffe build, GPU recommended | `pip install mediapipe` |
| Speed on laptop CPU | ~1 fps | real-time (~20–30 fps, lite model) |
| Licence | non-commercial research | Apache-2.0 |

**Status (verified Nov 2026 on Windows 11):** Python 3.9 / 3.11 / 3.12 / 3.13 /
3.14, `mediapipe 1.2.0`, `opencv-python 5.0.0.93`, `numpy 2.5.4` — install
≈ 30 s (~150 MB download), first `check_env.py` run ≈ 4 s.  Lite model
inference ≈ 7 ms/frame on CPU (≈ 145 fps without camera/display overhead).
Non-ASCII (Chinese) folder and file names are supported.

---

## 1 · Quick start (5 minutes) · 快速开始

```bash
# 0) Python 3.9-3.14 works; 3.12 or 3.13 is the recommended choice. Check:
python --version

# 1) get the code
git clone https://github.com/Junjie31/ieda3130-pose-demo.git
cd ieda3130-pose-demo

# 2) (recommended) virtual environment
python -m venv .venv
#   Windows:      .venv\Scripts\activate
#   macOS/Linux:  source .venv/bin/activate

# 3) install
python -m pip install -r requirements.txt

# 4) self-test (downloads the ~6 MB model, runs one inference, checks unicode
#    paths and the camera)
python check_env.py

# 5) run
python pose_demo.py --webcam                 # live skeleton + joint angles (s = save, q = quit)
python pose_demo.py --image my_photo.jpg     # one photo -> skeleton + measurement table
```

`python check_env.py` should end with:

```
Python  : 3.12.14  (Windows AMD64)
numpy   : OK  v2.5.4
cv2     : OK  v5.0.0
mediapipe: OK  v1.2.0
inference: OK  (poses found on blank image = 0, expected 0)
unicode  : OK  (photos in folders with Chinese names work)
camera 0 : not available (fine if you only use --image)
----------------------------------------------------------------
All good. Try:  python pose_demo.py --image your_photo.jpg
```

No camera / no admin rights? Open **`IEDA3130_Pose_Demo_Colab.ipynb`** in Google
Colab — same code, runs in the browser.

---

## 2 · What the demo outputs · 输出说明

Running `python pose_demo.py --image seated_side.jpg --known shoulder_elbow=33.5 --known-side R` prints

```
measure                               No.        px    world_cm   calib_cm
--------------------------------------------------------------------------
Shoulder-elbow length (L)              22      81.0        27.9       32.4
Elbow-fingertip length (L)             23     101.2        31.0       40.5
Shoulder-wrist length (proxy 25) (L)     25     157.1        51.5       62.9
Hip-knee length (proxy 13/14) (L)      13     116.7        35.7       46.7
Knee-ankle length (proxy 16) (L)       16     113.4        43.4       45.4
Shoulder above hip (vertical) (L)      10     151.0        50.9       60.4
Eye above hip (vertical) (L)            9     202.0        66.1       80.9
Elbow above hip (vertical) (L)         11     150.0        50.6       60.0
...
Shoulder-elbow length (R)              22      83.7        28.6       33.5
...
Joint angles (deg):
  elbow_flexion_L              172.1
  knee_flexion_L               175.3
  hip_flexion_L                120.7
  elbow_flexion_R              176.5
  knee_flexion_R               103.6
  hip_flexion_R                 95.1
  trunk_lean_deg                 2.2
  neck_forward_deg              45.0
[ok] image -> .../outputs/seated_side_pose.jpg
[ok] csv   -> .../outputs/seated_side_measures.csv
```

and saves `outputs/<name>_pose.jpg` (skeleton overlay) and
`outputs/<name>_measures.csv` (UTF-8, opens in Excel).

Three length columns, three different ideas:

| column | how it is obtained | when to trust it |
|---|---|---|
| `px` | Euclidean distance between two landmarks in the photo | always available; only comparable within one photo |
| `world_cm` | MediaPipe's 3-D *world landmarks* (metres, hip-centred) × 100 | no calibration needed, but learned from data → typically ±5–10 % |
| `calibrated_cm` | `px` × (tape-measured cm ÷ px) using the `--known` length(s) you supply | best when the body segment is parallel to the image plane (side/front view) |

### Landmark → Pheasant dimension mapping

| Pheasant no. (Lab 2 record sheet) | Dimension | Landmarks used | Caveat |
|---|---|---|---|
| 22 | Shoulder-elbow length | shoulder (11/12) → elbow (13/14) | good match |
| 23 | Elbow-fingertip length | elbow → index tip (19/20) | index landmark ≈ fingertip |
| 25 | Shoulder-grip length | shoulder → wrist (15/16) | add ≈ ½ hand length for the grip centre |
| 13 / 14 | Buttock-knee / buttock-popliteal | hip (23/24) → knee (25/26) | joint centre ≠ back of buttock → under-estimate |
| 16 | Popliteal height | knee → ankle (27/28) | add ankle-to-floor (~7 cm) and 2 cm shoe allowance |
| 9 / 10 / 11 | Sitting eye / shoulder / elbow height | vertical offset above hip | needs a true side view and a vertical upper arm |
| 12 | Thigh thickness | — | **not observable** from a skeleton (soft tissue) |

> Pip may print a "new release of pip is available" notice — harmless, ignore it.
> MediaPipe's C++ log lines are silenced by the demo; set `IEDA3130_VERBOSE=1`
> if you want to see them when debugging.

---

## 3 · Calibration with a tape measure · 标定

```
--known NAME=CM        one straight-line length you measured with the tape
--known-side L|R       which side that tape measurement belongs to (optional)
```

Valid `NAME`s (only straight-line measures can calibrate):
`shoulder_elbow`, `elbow_fingertip`, `shoulder_wrist`, `hip_knee`, `knee_ankle`.

```bash
python pose_demo.py --image side.jpg --known shoulder_elbow=33.5
python pose_demo.py --image side.jpg --known shoulder_elbow=33.5 --known-side R
python pose_demo.py --image side.jpg --known shoulder_elbow=33.5 --known elbow_fingertip=46
```

* In a **side-view** photo the far side is usually hidden, so one `--known` value
  is enough.
* In a **front-view** photo both arms are visible and both pixel lengths are
  used; if you measured only one arm, add `--known-side R` (or `L`) so the scale
  comes from that arm only.  Bad values (`abc`, `0`, `-5`, misspelled names) are
  rejected with a one-line error instead of a Python traceback.

---

## 4 · How to take a usable photo · 拍摄要求

1. **Side view**, camera at roughly hip height, 2–3 m away, lens parallel to the body plane.
2. Whole body in frame, plain background, no loose clothing hiding the joints.
3. For the seated (workstation) posture: sit as you would type — upper arm vertical, forearm horizontal.
4. Measure **one** segment with the tape (e.g. shoulder-elbow) and pass it with `--known shoulder_elbow=<cm>`.
5. Repeat for your lab partner, compare with your Lab 2 Excel sheet.

Flags: `--model full|heavy` (more accurate, slower) · `--label` (print landmark
indices 0-32) · `--no-show` (headless) · `--camera 1` (external webcam) ·
`--video clip.mp4` · `--known-side L|R`.

---

## 5 · Troubleshooting · 常见问题

| Symptom | Fix |
|---|---|
| `pip install mediapipe` finds no wheel | Use Python 3.9–3.14 (64-bit). mediapipe 1.2.0 ships a `py3-none-win_amd64` wheel plus macOS Apple-Silicon and Linux x64/ARM64 wheels. **Intel Mac**: `pip install "mediapipe<0.10.22"` with Python ≤ 3.12, or use the Colab notebook. |
| `ModuleNotFoundError: cv2` / `mediapipe` | Activate the virtual environment first, then `python -m pip install -r requirements.txt`. If you used `python -m venv`, the venv's `python` must be used to run the script. |
| Model download fails (campus Wi-Fi / mainland China) | Download the `.task` file in a browser from the URL printed on screen and save it to `models/pose_landmarker_lite.task` (or `_full`/`_heavy`). |
| Black window / `cannot open source 0` | macOS: System Settings → Privacy → Camera → allow Terminal/VS Code. Try `--camera 1`. Close Zoom/Teams which may hold the camera. |
| `cv2.imshow` error on Linux server / WSL / Colab | add `--no-show`; results are still saved in `outputs/` (the demo now tells you this instead of crashing). |
| Chinese characters in the folder/photo name | Supported (the demo reads/writes images through a Unicode-safe path and re-encodes console output as UTF-8). Very old PowerShell consoles may still *display* odd characters, but files are written correctly. |
| "No person detected" | whole body must be visible; improve lighting; try `--model full`. A warning is printed and an annotated image is still saved. |
| `[error] unknown --known key` | use one of the five valid straight-line names printed in the error message. |
| Very slow | use `--model lite`; close other apps; resize the video/photo. |
| Excel shows CSV rows in one column | Excel → Data → From Text/CSV → delimiter = comma, encoding = UTF-8. |
| Weird `W0000 ... inference_feedback_manager` lines | Only visible if you set `IEDA3130_VERBOSE=1`. They are harmless MediaPipe warnings; unset the variable to hide them again. |

---

## 6 · Files

```
pose_demo.py                     CLI: --image / --webcam / --video, drawing, CSV export
measure.py                       pure-numpy geometry: landmark -> Pheasant dimensions, joint angles
check_env.py                     one-shot environment self-test (model, inference, unicode, camera)
tests/test_measure.py            optional unit tests for measure.py (no camera/mediapipe needed)
requirements.txt                 mediapipe, opencv-python, numpy
IEDA3130_Pose_Demo_Colab.ipynb   browser fallback (Google Colab)
WORKSHEET.md                     one-page record sheet for the in-class exercise
LICENSE                          MIT (code); MediaPipe models are Apache-2.0
models/                          model bundle (auto-downloaded, git-ignored)
outputs/                         your results (git-ignored)
```

Optional unit test: `python tests/test_measure.py` → 20 tests, no camera needed.

---

## 7 · Discussion questions for your group · 讨论题

1. For each dimension, is the camera estimate **systematically** higher or lower than the tape? Why (joint centre vs. bony landmark, soft tissue, perspective)?
2. Which Lab 2 dimensions can **never** be obtained from a 2-D skeleton? What extra sensor would you need?
3. BS 7310 asks for the elbow ≈ 90° at keyboard height. Use `--webcam`, adjust your chair, and record the chair height at which `elbow_flexion` ≈ 90° and `knee_flexion` ≈ 90°. Does it agree with your Step 8c answer?
4. If an employer wanted to use this camera method for the Ch. 509B risk assessment of 200 operators, what are the **ethical / privacy** issues compared with the tape measurement + consent form you used in Lab 2?
5. Estimate the measurement error (mean ± SD of `calibrated_cm − tape_cm`) over your group. Is it small enough for the *Extreme Case Approach* used in Step 8d?

## References
- Pheasant, S. (1986). *Bodyspace: Anthropometry, Ergonomics and Design*. Table 4.36 (Hong Kong Chinese adults).
- BS 7310:1990 – Ergonomics of the design of VDT workstations.
- Cao, Z. et al. (2017). *Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields* (OpenPose), CVPR.
- Bazarevsky, V. et al. (2020). *BlazePose: On-device Real-time Body Pose Tracking*. Google AI.
- MediaPipe Pose Landmarker docs: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python

*Licence: code MIT (see LICENSE); MediaPipe models Apache-2.0. Teaching material of IEDA 3130, HKUST – do not redistribute student photos.*
