#!/usr/bin/env python3
"""Run this first:  python check_env.py
It checks Python/package versions, downloads the model, and runs one inference
on a synthetic image so you know the whole chain works BEFORE the lab session."""
import platform
import sys

print(f"Python  : {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")
if sys.version_info < (3, 9):
    sys.exit("[fail] Python >= 3.9 is required by mediapipe. Install 3.11 or 3.12 (recommended).")

for pkg in ("numpy", "cv2", "mediapipe"):
    try:
        m = __import__(pkg)
        print(f"{pkg:<8}: OK  v{getattr(m, '__version__', '?')}")
    except Exception as e:
        sys.exit(f"[fail] cannot import {pkg}: {e}\n       -> python -m pip install -r requirements.txt")

import numpy as np
import cv2
import pose_demo

model = pose_demo.ensure_model("lite")
img = np.full((480, 640, 3), 128, dtype=np.uint8)
with pose_demo.build_landmarker(model, "image") as lm:
    res = lm.detect(pose_demo.to_mp_image(img))
print(f"inference: OK  (poses found on blank image = {len(res.pose_landmarks)}, expected 0)")

cap = cv2.VideoCapture(0)
if cap.isOpened():
    ok, _ = cap.read()
    print(f"camera 0 : {'OK' if ok else 'opened but no frame (check permission)'}")
    cap.release()
else:
    print("camera 0 : not available (fine if you only use --image)")
print("\nAll good. Try:  python pose_demo.py --image your_photo.jpg")
