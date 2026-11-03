# IEDA 3130 · Pose Estimation Demo (OpenPose-style, built on MediaPipe)

> HKUST · Department of Industrial Engineering and Decision Analytics  
> IEDA 3130 Ergonomics and Safety Management · Fall 2026  
> Instructor: Prof. Richard SO · TA: Junjie HOU

This demo shows how a camera + a pose-estimation network can extract the body
landmarks that we measured **by hand with an anthropometer in Lab 2**
(Anthropometry: Workstation Design), and how to judge a seated posture against
the **BS 7310** criterion (elbow ≈ 90°, forearm horizontal at keyboard height).

It is intentionally small: **4 short Python files, 3 direct pip requirements, no
GPU, no compiler** (the 4th file is an optional unit test; pip installs ≈ 20
packages / ≈ 350 MB of disk transitively, all pre-built wheels). Everything runs
on a normal laptop (Windows / macOS / Linux) or in Google Colab.

| | OpenPose (CMU, 2017) | This demo (MediaPipe Pose Landmarker) |
|---|---|---|
| Keypoints | 18 / 25 body (+hands/face) | 33 body landmarks, 2-D + 3-D "world" coords |
| Install | C++/CUDA/Caffe build, GPU recommended | `pip install mediapipe` |
| Speed on laptop CPU | ~1 fps | real-time (~20–30 fps, lite model) |
| Licence | non-commercial research | Apache-2.0 |

**Status (verified Nov 2026 on Windows 11):** Python 3.9 / 3.11 / 3.12 / 3.13 /
3.14, `mediapipe 1.2.0`, `opencv-contrib-python 5.0.0.93`, `numpy 2.5.4` —
install ≈ 30 s, first `check_env.py` run ≈ 4 s.  Lite model inference ≈ 7 ms/frame
on CPU (≈ 145 fps without camera/display overhead); the live demo was observed at
≈ 20–30 fps end-to-end.  Non-ASCII (Chinese) folder, file and Windows-profile paths are
supported — the demo automatically uses an ASCII temporary working directory around MediaPipe
to avoid a Windows native-library limitation (see Troubleshooting).
**Verified on Windows 11**; macOS is supported by design (one-click `.command` scripts and a
Route A-macOS section are included) but could not be executed on the build machine, and no
real webcam was available here — please run `check_env.py` on your own laptop and report any
deviation. Intel-Mac mediapipe wheels are a known risk (see Troubleshooting).

---

## 1 · Quick start · 快速开始

**No Python experience assumed.** Pick one of the three routes below.

### Route A — Complete beginner: Anaconda + Git (recommended)

This is the same tool chain used in IEDA2100S Lab 8 (Object Detection).

**Step 0 · Install Anaconda** (skip if already installed)

1. Go to <https://www.anaconda.com/download> and download the **Windows 64-bit** installer
   (macOS/Linux: pick your installer).
2. Run it and accept the default options. It installs the **Anaconda Prompt**, which is the
   terminal you will use from now on.

**Step 0b · Install Git** (skip if already installed)

1. Go to <https://git-scm.com/download/win> and download the 64-bit **Standalone Installer**.
2. Run it, accept the defaults, then open a **new** Anaconda Prompt.
3. Check: `git --version`

**Step 1 · Open the terminal**

Press Start → search **Anaconda Prompt** → open it.

> **Why does my normal terminal not know `conda` / `python`?**
> Windows does not put Anaconda on the system PATH (on purpose — it would clash with other
> Python installs), so in a plain **Command Prompt / PowerShell / VS Code terminal** you may see
> `'conda' is not recognized` or `'python' was not found` (or the Microsoft Store opens).
> Use one of these:
>
> 1. **Anaconda Prompt** (Start menu) — the recommended terminal, everything works there; or
> 2. double-click **`open_shell.bat`** in the project folder — it opens a Command Prompt with
>    the `ieda3130` environment already active, so you can type `python …` / `conda …` directly; or
> 3. enable conda in your own terminal **once**, from the Anaconda Prompt:
>
>    ```bat
>    conda init cmd.exe          REM makes conda work in Command Prompt
>    conda init powershell       REM makes conda work in PowerShell / VS Code
>    ```
>
>    then **close and reopen** that terminal and run `conda activate ieda3130`.
>
> Do **not** add Anaconda to the Windows PATH manually; `conda init` is the supported way.
> Check with `where conda`, `conda --version`, `python --version`.

**Step 2 · Get the code**

```bat
git clone https://github.com/Junjie31/ieda3130-pose-demo.git
cd ieda3130-pose-demo
```

**Step 3 · Create and activate the environment**

```bat
conda create -n ieda3130 python=3.12 -y
conda activate ieda3130
```

(The `ieda3130` environment is an isolated "kitchen" with its own Python 3.12 — it cannot
break your other projects.)

**Step 4 · Install the packages** (about 30 s, ~350 MB disk, no compiler needed)

```bat
python -m pip install -r requirements.txt
```

**Step 5 · Self-test** (downloads the pose model, ~6 MB)

```bat
python check_env.py
```

**Step 6 · Run the demo**

```bat
REM live camera demo - press s to save a snapshot, f for fullscreen, q to quit
python pose_demo.py --webcam

REM one photo -> skeleton + measurement table
python pose_demo.py --image my_photo.jpg
```

**Even simpler: one-click scripts.** Double-click `setup_windows.bat` once (it finds Anaconda,
creates the environment, installs everything and runs the self-test), then:

- `run_webcam.bat` — live camera demo
- drag a photo file onto `run_image.bat` — single-photo measurement
- `open_shell.bat` — opens a Command Prompt with the `ieda3130` environment active, for when
  you want to type `python …` / `conda …` commands yourself (this is the answer to
  "`conda` is not recognized" in a normal terminal)

`environment.yml` is also provided if you prefer `conda env create -f environment.yml`.

### Route A-macOS — Complete beginner on a Mac

macOS does **not** ship Python or Conda by default, so start here just like on Windows.

**Step 0 · Install Anaconda**

1. Go to <https://www.anaconda.com/download> and download the **macOS installer**.
2. Pick **Apple Silicon** if your Mac has an M1/M2/M3/M4 chip (Apple menu → About This Mac →
   "Chip: Apple M…"), otherwise pick **Intel**.
3. Double-click the downloaded `.pkg` and accept the default options.

**Step 0b · Install Git**

1. Open **Terminal** (Spotlight search → "Terminal").
2. Type `git --version`.
   - If Git is already present, you see a version number.
   - If not, macOS offers to install the **Command Line Developer Tools**: click **Install**,
     then run `git --version` again. (Alternative: <https://git-scm.com/download/mac>.)

**Step 1 · Get Conda into your shell**

The Anaconda installer normally runs `conda init` for you. If `conda --version` says
"command not found", run this once, then close and reopen Terminal:

```bash
# per-user Anaconda (the default .pkg location)
~/anaconda3/bin/conda init zsh
# other common locations - use whichever exists:
#   /opt/anaconda3/bin/conda init zsh          (system-wide Anaconda install)
#   ~/opt/anaconda3/bin/conda init zsh         (Miniconda "opt" install)
#   ~/miniconda3/bin/conda init zsh            (Miniconda per-user install)
```

**Step 2 · Get the code**

```bash
git clone https://github.com/Junjie31/ieda3130-pose-demo.git
cd ieda3130-pose-demo
```

**Step 3 · Create and activate the environment**

```bash
conda create -n ieda3130 python=3.12 -y
conda activate ieda3130
```

**Step 4 · Install the packages** (about 30 s, ~350 MB disk, no compiler needed)

```bash
python -m pip install -r requirements.txt
```

**Step 5 · Self-test** (downloads the pose model, ~6 MB)

```bash
python check_env.py
```

**Step 6 · Run the demo**

```bash
python pose_demo.py --webcam               # live skeleton + posture feedback (s = save, q = quit)
python pose_demo.py --image my_photo.jpg   # one photo -> skeleton + measurement table
```

**Camera permission on macOS (important):** the first time the live demo runs, macOS asks
whether Terminal may use the camera — click **Allow**. If you missed it, enable
**System Settings → Privacy & Security → Camera → Terminal**.

**Even simpler: one-click scripts.** Double-click `setup_mac.command` once, then
`run_webcam.command` or drag a photo onto `run_image.command`. If macOS refuses to open a
`.command` file ("unidentified developer"), right-click → **Open**, or run
`bash setup_mac.command` in Terminal. You may need once: `chmod +x *.command`.

> **Apple Silicon vs Intel:** MediaPipe publishes Apple-Silicon wheels, so M-series Macs are
> the well-supported path. On older Intel Macs `pip install mediapipe` may find no wheel —
> try `python -m pip install "mediapipe<0.10.22"`; if that fails, use the Colab notebook.
> (macOS itself was **not** available on the machine that tested this demo — see the
> "verified" note at the top; please report anything that does not match these steps.)

### Route B — Python is already installed (venv route)

```bash
# Python 3.9-3.14 works; 3.12 or 3.13 recommended
python --version
git clone https://github.com/Junjie31/ieda3130-pose-demo.git
cd ieda3130-pose-demo
python -m venv .venv
#   Windows:      .venv\Scripts\activate
#   macOS/Linux:  source .venv/bin/activate
python -m pip install -r requirements.txt
python check_env.py
python pose_demo.py --webcam
```

### Route C — No install at all

Open **`IEDA3130_Pose_Demo_Colab.ipynb`** in Google Colab — same code in the browser
(photo mode only; Colab cannot stream your camera).

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

### What you see in the live window (`--webcam`)

The window is **resizable**: drag its corner, maximise it, or press `f` for fullscreen — the
video is automatically scaled (letterboxed, aspect ratio preserved) to the window size.

| HUD line | Meaning |
|---|---|
| `FPS 24.1  model=lite  [s]=save [f]=full [q]=quit` | frames per second and keyboard shortcuts |
| `POSTURE OK` / `ADJUST (4/6 OK)` | how many of the visible angles are inside the targets |
| `elbow_L 92.3 deg OK` (green) | elbow internal angle ~90° (80–100°) |
| `knee_R 101.2 deg adjust` (orange) | knee internal angle outside 80–100° |
| `trunk_lean 2.4 deg OK` | trunk lean from vertical < 10° |
| `neck_fwd 25.4 deg adjust` | head-forward angle < 20° |

Green = inside the BS 7310-inspired target, orange = adjust your chair/keyboard, grey = that
landmark is not visible enough. These are teaching guidelines, not a pass/fail assessment.
The HUD is deliberately compact and scales with the frame size (about a fifth of the picture),
so it does not hide the subject.
Keys: `s` saves a snapshot (skeleton + HUD + CSVs), `f` toggles fullscreen, `q` / `Esc` /
closing the window stops the camera. The photo mode window has the same resizing/fullscreen
behaviour and closes on any key (other than `f`).

---

## 2 · What the demo outputs · 输出说明

Running `python pose_demo.py --image seated_side.jpg --known shoulder_elbow=33.5 --known-side R` prints
(real output; trimmed here by a few rows)

```
measure                                 No.        px    world_cm   calib_cm
----------------------------------------------------------------------------
Shoulder-elbow length (L)                22      81.0        27.9       32.4
Elbow-fingertip length (L)               23     101.2        31.0       40.5
Shoulder-wrist length (proxy 25) (L)       25     157.1        51.5       62.9
Hip-knee length (proxy 13/14) (L)        13     116.7        35.7       46.7
Knee-ankle length (proxy 16) (L)         16     113.4        43.4       45.4
Shoulder above hip (vertical) (L)        10     151.0        50.9       60.4
Eye above hip (vertical) (L)              9     202.0        66.1       80.9
Elbow above hip (vertical) (L)           11     150.0        50.6       60.0
Shoulder-elbow length (R)                22      83.7        28.6       33.5
Elbow-fingertip length (R)               23      95.4        31.8       38.2
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
[ok] image    -> <project>/outputs/seated_side_pose.jpg
[ok] measures -> <project>/outputs/seated_side_measures.csv
[ok] angles   -> <project>/outputs/seated_side_angles.csv
```

Rows whose landmarks are not visible enough (visibility < 0.5) are marked with
`!` after the measure name and a warning below the table — **never record those
rows**. `-` means "cannot be computed/calibrated", not zero.

The demo always writes into `<project>/outputs/` (next to the code, not next to
your shell):
`<name>_pose.jpg` (skeleton overlay), `<name>_measures.csv` (length table) and
`<name>_angles.csv` (joint angles). All three are UTF-8 and open in Excel.

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
  is enough and the taped segment comes out exactly at your tape value.
* In a **front-view** photo both arms are visible. Without `--known-side` the
  two sides' pixel lengths are averaged before the cm/px ratio is formed, so the
  taped segment will be close to but not exactly your tape value. If you
  measured only one arm, always add `--known-side R` (or `L`) so the scale comes
  from that arm only.
* With several `--known` values the mean cm/px ratio over the provided segments
  is used; more values → steadier scale.
* Bad input (`abc`, `0`, `-5`, `999`, misspelled names) is rejected with a
  one-line error and exit code 1 instead of a Python traceback.
* Calibration only works for segments that are roughly parallel to the image
  plane in the photo (side/front view); a foreshortened limb gives a wrong
  scale.

---

## 4 · How to take a usable photo · 拍摄要求

1. **Side view**, camera at roughly hip height, 2–3 m away, lens parallel to the body plane.
2. Whole body in frame, plain background, no loose clothing hiding the joints.
3. For the seated (workstation) posture: sit as you would type — upper arm vertical, forearm horizontal.
4. Measure **one** segment with the tape (e.g. shoulder-elbow) and pass it with `--known shoulder_elbow=<cm>`.
5. Repeat for your lab partner, compare with your Lab 2 Excel sheet.

Flags: `--model full|heavy` (more accurate, slower) · `--label` (print landmark
indices 0-32) · `--no-show` (headless) · `--camera 1` (external webcam) ·
`--video clip.mp4` · `--known-side L|R` · `--save-every N` / `--max-frames N`
(video/webcam: auto-save every Nth frame, stop after N frames — useful headless).

---

## 5 · Troubleshooting · 常见问题

| Symptom | Fix |
|---|---|
| `pip install mediapipe` finds no wheel | Use Python 3.9–3.14 (64-bit). mediapipe 1.2.0 ships a `py3-none-win_amd64` wheel plus macOS Apple-Silicon and Linux x64/ARM64 wheels. **Intel Mac**: `pip install "mediapipe<0.10.22"` with Python ≤ 3.12 (this workaround was NOT verified on this machine), or use the Colab notebook. |
| `ModuleNotFoundError: cv2` / `mediapipe` | You are running a different Python than the one where the demo is installed (the conda environment is not active). Use the **Anaconda Prompt** with `conda activate ieda3130`, double-click **`open_shell.bat`**, or run `conda init cmd.exe` / `conda init powershell` once and reopen your terminal. |
| `'conda' is not recognized` in Command Prompt / PowerShell / VS Code | Normal on Windows: Anaconda is not on the system PATH. Use the **Anaconda Prompt** (Start menu), double-click **`open_shell.bat`**, or enable it once from Anaconda Prompt: `conda init cmd.exe` (Command Prompt) and/or `conda init powershell` (PowerShell), then close and reopen that terminal and run `conda activate ieda3130`. Do not edit PATH manually. |
| `python` opens the Microsoft Store, or `'python' was not found` | No Python is on the PATH (or the Store alias grabs it). Use the Anaconda Prompt / `open_shell.bat`, or turn off the alias: Settings → Apps → Advanced app settings → App execution aliases → switch off `python.exe` / `python3.exe`, then reopen the terminal. |
| `conda activate` says "conda not initialized for this shell" | Run the matching one-time init in the Anaconda Prompt: `conda init cmd.exe`, `conda init powershell`, or `conda init bash` (Git Bash), then reopen the terminal. |
| `where python` prints several unrelated paths | Your terminal is using another Python. With the environment active, `where python` should list first `…\anaconda3\envs\ieda3130\python.exe`. Use `open_shell.bat` if unsure. |
| setup script says Conda not found | Install Anaconda from <https://www.anaconda.com/download> (default options), then **close and reopen** the Anaconda Prompt / run `setup_windows.bat` again. |
| **macOS**: `conda: command not found` | Run `~/anaconda3/bin/conda init zsh` once (use `/opt/anaconda3` or `~/opt/anaconda3` for other install locations), then close and reopen Terminal. |
| **macOS**: camera window stays black / no camera | System Settings → Privacy & Security → Camera → enable **Terminal**, then restart Terminal. |
| **macOS**: cannot open `setup_mac.command` ("unidentified developer") | Right-click the file → **Open**, or run `bash setup_mac.command` in Terminal; once: `chmod +x *.command`. |
| **macOS Intel**: `pip install mediapipe` finds no wheel | `python -m pip install "mediapipe<0.10.22"` with Python ≤ 3.12; if that fails use the Colab notebook (Apple-Silicon wheels are the well-supported path). |
| One-click script says environment "ieda3130" not found | Run `setup_windows.bat` / `setup_mac.command` first (it creates the environment), or create it manually: `conda create -n ieda3130 python=3.12 -y`. |
| `run_image.bat` closes immediately | Drag the photo **onto** the `.bat` icon (passing the file as an argument); if you double-click it with no photo it prints usage and exits. |
| Model download fails (campus Wi-Fi / mainland China) | The download shows a percentage but times out after 120 s. Download the `.task` file in a browser from the URL printed on screen and save it to `models/pose_landmarker_lite.task` (or `_full`/`_heavy`). |
| Black window / `cannot open camera 0` | Try `--camera 0` and `--camera 1`; close Zoom/Teams/Camera; Windows Settings → Privacy → Camera / macOS System Settings → Privacy → allow Terminal or VS Code. |
| `cv2.imshow` error on Linux server / WSL / Colab | add `--no-show`; for a **photo** the annotated image and CSVs are still saved in `outputs/`. For `--video`/`--webcam` add `--save-every 15` to save frames headlessly (nothing is saved otherwise). |
| Chinese characters anywhere in the path (photo, project folder, **or Windows user profile**) | Supported. Images use a Unicode-safe read/write path; console output is UTF-8; when the project sits in a non-ASCII folder the model is copied to an ASCII temp path; and when a **parent** of the working directory is non-ASCII (e.g. `C:\Users\陳大文\Desktop\...`) MediaPipe is run from an ASCII temp working directory, which avoids a Windows native-library crash. You will see two `[info]` lines about this on the first run — that is expected. |
| Window shows the video small in the top-left corner after maximising (old versions) | Fixed: the live/photo windows are resizable and the picture now scales to the window size (aspect ratio preserved, with letterbox bars). Press `f` for fullscreen. |
| `check_env.py` exits silently with a negative code (e.g. `-1073740791` / `0xC0000409`), no message | This is the MediaPipe non-ASCII-cwd crash described above; clearly print the path (`pwd` / `cd`), move the project to `C:\IEDA3130` if the automatic ASCII-work-dir fallback cannot be used, or run the command from an ASCII folder with absolute paths. |
| "No person detected" | whole body must be visible; improve lighting; try `--model full`. A warning is printed and an annotated image is still saved. |
| Rows marked `!` in the table | at least one landmark had visibility < 0.5 (e.g. legs behind a desk). Do not record those values; re-take the photo. |
| `[error] unknown --known key` | use one of the five valid straight-line names printed in the error message. |
| Very slow | use `--model lite`; close other apps; resize the video/photo. |
| Excel shows CSV rows in one column | Excel → Data → From Text/CSV → delimiter = comma, encoding = UTF-8. |
| Weird `W0000 ... inference_feedback_manager` lines | Only visible if you set `IEDA3130_VERBOSE=1`. They are harmless MediaPipe warnings; unset the variable to hide them again. |
| Cannot find `outputs/` | Results always go to the `outputs/` folder **next to `pose_demo.py`** (not the folder your shell is in). |

---

## 6 · Files

```
pose_demo.py                     CLI: --image / --webcam / --video, live HUD, drawing, CSV export
measure.py                       pure-numpy geometry: landmark -> Pheasant dimensions, joint angles
check_env.py                     one-shot environment self-test (model, inference, unicode, camera)
tests/test_measure.py            unit tests for measure.py (no camera/mediapipe needed)
tests/test_stream_smoke.py       simulated live-camera test (no webcam hardware needed)
requirements.txt                 mediapipe, opencv-contrib-python, numpy
environment.yml                  optional Conda environment (one-command install)
setup_windows.bat                one-click Windows setup (finds Anaconda, installs everything)
run_webcam.bat                   one-click Windows live-camera launcher
run_image.bat                    one-click Windows single-photo launcher (drag a photo onto it)
open_shell.bat                   opens a Command Prompt with the ieda3130 environment active
setup_mac.command                one-click macOS setup (finds Anaconda, installs everything)
run_webcam.command               one-click macOS live-camera launcher
run_image.command                one-click macOS single-photo launcher (drag a photo onto it)
IEDA3130_Pose_Demo_Colab.ipynb   browser fallback (Google Colab)
WORKSHEET.md                     one-page record sheet for the in-class exercise
LICENSE                          MIT (code); MediaPipe models are Apache-2.0
samples/sample_pose.jpg          sample photo if you do not have one yet
models/                          model bundle (auto-downloaded, git-ignored)
outputs/                         your results: <name>_pose.jpg, _measures.csv, _angles.csv (git-ignored)
```

Optional checks: `python tests/test_measure.py` (20 geometry tests, no camera) and
`python tests/test_stream_smoke.py` (runs the live `--webcam` code path on a simulated
camera, no hardware needed).

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
- `samples/sample_pose.jpg` is from Google's MediaPipe sample assets (Apache-2.0).

*Licence: code MIT (see LICENSE); MediaPipe models Apache-2.0. Teaching material of IEDA 3130, HKUST – do not redistribute student photos.*
