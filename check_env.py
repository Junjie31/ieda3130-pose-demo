#!/usr/bin/env python3
"""Run this first:  python check_env.py

It checks Python/package versions, downloads the pose model, runs one inference
on a synthetic image, tests that non-ASCII (e.g. Chinese) file paths work, and
reports whether a camera is available -- all BEFORE the lab session."""
import os
import platform
import shutil
import sys
import tempfile

# Keep MediaPipe / TensorFlow-Lite C++ logging quiet (must be set before import).
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("GLOG_logtostderr", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Make printing safe for non-ASCII paths on a legacy Windows console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

print("=" * 64)
print(" IEDA 3130 pose demo - environment self-test")
print("=" * 64)
print(f"Python  : {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")

if sys.version_info < (3, 9):
    sys.exit("[fail] Python >= 3.9 is required by mediapipe.\n"
             "       Install Python 3.12 or 3.13 (recommended) from python.org.")
if sys.version_info >= (3, 15):
    print("[warn] Python 3.15+ has not been tested with this demo; Python 3.12/3.13 "
          "is the recommended choice.")

for pkg in ("numpy", "cv2", "mediapipe"):
    try:
        m = __import__(pkg)
        print(f"{pkg:<8}: OK  v{getattr(m, '__version__', '?')}")
    except Exception as e:
        sys.exit(f"[fail] cannot import {pkg}: {e}\n"
                 f"       -> python -m pip install -r requirements.txt\n"
                 f"       (make sure your virtual environment is activated)")

import numpy as np
import cv2
import pose_demo  # noqa: E402  (imports mediapipe after setting the log env vars)

# --------------------------------------------------------------------------- #
# 1) model download + one inference
# --------------------------------------------------------------------------- #
try:
    model = pose_demo.ensure_model("lite")
except SystemExit as e:  # model download failed
    print(e)
    sys.exit(1)
img = np.full((480, 640, 3), 128, dtype=np.uint8)
try:
    with pose_demo.build_landmarker(model, "image") as lm:
        res = pose_demo.detect_image(lm, img)
except Exception as e:
    sys.exit(f"[fail] inference failed: {e}")
print(f"inference: OK  (poses found on blank image = {len(res.pose_landmarks)}, expected 0)")

# --------------------------------------------------------------------------- #
# 2) non-ASCII paths (Chinese folder / file names are common on student laptops)
# --------------------------------------------------------------------------- #
tmpdir = tempfile.mkdtemp(prefix="ieda-check-")
try:
    uni_dir = os.path.join(tmpdir, "路径 测试")
    os.makedirs(uni_dir, exist_ok=True)
    uni_path = os.path.join(uni_dir, "照片.jpg")
    ok_write = pose_demo.imwrite_unicode(uni_path, img)
    back = pose_demo.imread_unicode(uni_path) if ok_write else None
    if ok_write and back is not None and back.shape == img.shape:
        print("unicode  : OK  (photos in folders with Chinese names work)")
    else:
        print("unicode  : WARN - could not read/write a file with a non-ASCII path here;\n"
              "           keep the project folder and photo paths in plain ASCII (e.g. C:\\IEDA3130)")
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

# --------------------------------------------------------------------------- #
# 3) camera
# --------------------------------------------------------------------------- #
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ok, _ = cap.read()
    print(f"camera 0 : {'OK' if ok else 'opened but no frame (check camera permission)'}")
    cap.release()
else:
    print("camera 0 : not available (fine if you only use --image)")

print("-" * 64)
print("All good. Try:  python pose_demo.py --image your_photo.jpg")
print("Live camera  :  python pose_demo.py --webcam        (s = save, q = quit)")
print("No camera?   :  use the Google Colab notebook instead.")
