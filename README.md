# IEDA 3130 · Pose Estimation Demo (OpenPose-style, built on MediaPipe)

> HKUST · Department of Industrial Engineering and Decision Analytics  
> IEDA 3130 Ergonomics and Safety Management · Fall 2026  
> Instructor: Prof. Richard SO · TA: Junjie HOU

This demo shows how a camera + a pose-estimation network can extract the body
landmarks that we measured **by hand with an anthropometer in Lab 2**
(Anthropometry: Workstation Design), and how to judge a seated posture against
the **BS 7310** criterion (elbow ≈ 90°, forearm horizontal at keyboard height).

It is intentionally small: **3 Python files, 3 pip packages, no GPU, no compilation.**
Everything runs on a normal laptop (Windows / macOS / Linux) or in Google Colab.

| | OpenPose (CMU, 2017) | This demo (MediaPipe Pose Landmarker) |
|---|---|---|
| Keypoints | 18 / 25 body (+hands/face) | 33 body landmarks, 2-D + 3-D "world" coords |
| Install | C++/CUDA/Caffe build, GPU recommended | `pip install mediapipe` |
| Speed on laptop CPU | ~1 fps | real-time (~20–30 fps, lite model) |
| Licence | non-commercial research | Apache-2.0 |

---

## 1 · Quick start (5 minutes) · 快速开始

```bash
# 0) Python 3.9–3.12 recommended (3.11 is the safest). Check:
python --version

# 1) get the code
git clone https://github.com/Junjie31/ieda3130-pose-demo.git
cd ieda3130-pose-demo

# 2) (recommended) virtual environment
python -m venv .venv
#   Windows:  .venv\Scripts\activate
#   macOS/Linux:  source .venv/bin/activate

# 3) install
python -m pip install -r requirements.txt

# 4) self-test (downloads the 5 MB model, runs one inference, checks camera)
python check_env.py

# 5) run
python pose_demo.py --webcam                 # live skeleton + joint angles (s = save, q = quit)
python pose_demo.py --image my_photo.jpg     # one photo -> skeleton + measurement table
```

No camera / no admin rights? Open **`IEDA3130_Pose_Demo_Colab.ipynb`** in Google Colab — same code, runs in the browser.

---

## 2 · What the demo outputs · 输出说明

Running `python pose_demo.py --image seated_side.jpg --known shoulder_elbow=33.5` prints

```
measure                              No.       px   world_cm     calib_cm
----------------------------------------------------------------------------
Shoulder-elbow length (R)             22    131.2       33.9         33.5
Elbow-fingertip length (R)            23    181.0       45.1         46.2
Shoulder-wrist length (proxy 25) (R)  25    178.3       44.6         45.5
Hip-knee length (proxy 13/14) (R)     13    170.4       42.0         43.5
Knee-ankle length (proxy 16) (R)      16    169.0       41.6         43.1
...
Joint angles (deg):
  elbow_flexion_R   91.3     knee_flexion_R   88.7     trunk_lean_deg  4.2
```

and saves `outputs/<name>_pose.jpg` (skeleton overlay) and `outputs/<name>_measures.csv`.

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

---

## 3 · How to take a usable photo · 拍摄要求

1. **Side view**, camera at roughly hip height, 2–3 m away, lens parallel to the body plane.
2. Whole body in frame, plain background, no loose clothing hiding the joints.
3. For the seated (workstation) posture: sit as you would type — upper arm vertical, forearm horizontal.
4. Measure **one** segment with the tape (e.g. shoulder-elbow) and pass it with `--known shoulder_elbow=<cm>`.
5. Repeat for your lab partner, compare with your Lab 2 Excel sheet.

Flags: `--model full|heavy` (more accurate, slower) · `--label` (print landmark indices 0-32) · `--no-show` (headless) · `--camera 1` (external webcam) · `--video clip.mp4`.

---

## 4 · Discussion questions for your group · 讨论题

1. For each dimension, is the camera estimate **systematically** higher or lower than the tape? Why (joint centre vs. bony landmark, soft tissue, perspective)?
2. Which Lab 2 dimensions can **never** be obtained from a 2-D skeleton? What extra sensor would you need?
3. BS 7310 asks for the elbow ≈ 90° at keyboard height. Use `--webcam`, adjust your chair, and record the chair height at which `elbow_flexion` ≈ 90° and `knee_flexion` ≈ 90°. Does it agree with your Step 8c answer?
4. If an employer wanted to use this camera method for the Ch. 509B risk assessment of 200 operators, what are the **ethical / privacy** issues compared with the tape measurement + consent form you used in Lab 2?
5. Estimate the measurement error (mean ± SD of `calibrated_cm − tape_cm`) over your group. Is it small enough for the *Extreme Case Approach* used in Step 8d?

---

## 5 · Troubleshooting · 常见问题

| Symptom | Fix |
|---|---|
| `pip install mediapipe` finds no wheel | Use Python 3.9–3.12 (64-bit). mediapipe 1.x ships wheels for Windows x64/ARM64, macOS **Apple Silicon**, Linux x64/ARM64. **Intel Mac**: `pip install "mediapipe<0.10.22"` with Python ≤ 3.12. Still stuck → use the Colab notebook. |
| `ModuleNotFoundError: cv2` | `python -m pip install opencv-python` (make sure the venv is activated) |
| Model download fails (campus Wi-Fi / China mainland) | Download the `.task` file in a browser from the URL printed on screen and save it to `models/pose_landmarker_lite.task` |
| Black window / `cannot open source 0` | macOS: System Settings → Privacy → Camera → allow Terminal/VS Code. Try `--camera 1`. Close Zoom/Teams which may hold the camera. |
| `cv2.imshow` error on Linux server / WSL / Colab | add `--no-show`; results are still saved in `outputs/` |
| "No person detected" | whole body must be visible; improve lighting; try `--model full` |
| Very slow | use `--model lite`; close other apps; resize video |

---

## 6 · Files

```
pose_demo.py                     CLI: --image / --webcam / --video, drawing, CSV export
measure.py                       pure-numpy geometry: landmark → Pheasant dimensions, joint angles
check_env.py                     one-shot environment self-test
requirements.txt                 mediapipe, opencv-python, numpy
IEDA3130_Pose_Demo_Colab.ipynb   browser fallback (Google Colab)
WORKSHEET.md                     one-page record sheet for the in-class exercise
models/                          model bundle (auto-downloaded, git-ignored)
outputs/                         your results (git-ignored)
```

## References
- Pheasant, S. (1986). *Bodyspace: Anthropometry, Ergonomics and Design*. Table 4.36 (Hong Kong Chinese adults).
- BS 7310:1990 – Ergonomics of the design of VDT workstations.
- Cao, Z. et al. (2017). *Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields* (OpenPose), CVPR.
- Bazarevsky, V. et al. (2020). *BlazePose: On-device Real-time Body Pose Tracking*. Google AI.
- MediaPipe Pose Landmarker docs: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python

*Licence: code MIT; MediaPipe models Apache-2.0. Teaching material of IEDA 3130, HKUST – do not redistribute student photos.*
