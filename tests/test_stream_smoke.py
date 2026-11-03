#!/usr/bin/env python3
"""Smoke test for the LIVE (webcam) branch -- no camera hardware required.

It monkey-patches ``cv2.VideoCapture`` so the same code path that runs when you
use ``--webcam`` (mirroring, virtual 33 ms timestamps, live posture HUD,
``--save-every`` saving, graceful end-of-stream) is exercised on frames taken
from ``samples/sample_pose.jpg``.

Run:   python tests/test_stream_smoke.py
Needs mediapipe + opencv (unlike tests/test_measure.py, which needs only numpy).
"""
import os
import shutil
import sys
import tempfile
import types

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import cv2  # noqa: E402

import pose_demo  # noqa: E402

SAMPLE = os.path.join(REPO, "samples", "sample_pose.jpg")


def test_posture_status_rules():
    """Pure logic: green/orange HUD decisions follow the documented targets."""
    angles = {
        "elbow_flexion_L": 92.0,    # OK (90 +/- 10)
        "elbow_flexion_R": 140.0,   # adjust
        "knee_flexion_L": 88.0,     # OK
        "knee_flexion_R": None,     # not visible -> grey line, not counted
        "trunk_lean_deg": 3.0,      # OK
        "neck_forward_deg": 45.0,   # adjust (> 20)
    }
    lines, n_ok, n_checked = pose_demo.posture_status(angles)
    assert n_checked == 5, n_checked          # 4 angles numeric + neck numeric
    assert n_ok == 3, n_ok                    # elbow L, knee L, trunk
    text = " | ".join(t for t, _ in lines)
    assert "adjust" in text and "OK" in text, text
    assert any("elbow_R" in t and "adjust" in t for t, _ in lines), text
    # a completely empty angles dict must not crash
    lines, n_ok, n_checked = pose_demo.posture_status({})
    assert (n_ok, n_checked) == (0, 0)


def test_live_branch_on_fake_camera():
    """Run pose_demo.run_stream() in '--webcam --no-show --save-every' mode."""
    if not os.path.exists(SAMPLE):
        print("SKIP: samples/sample_pose.jpg is missing")
        return
    img = cv2.imread(SAMPLE)
    assert img is not None, SAMPLE
    frames = [img.copy() for _ in range(6)]

    class FakeCapture:
        def __init__(self, src):
            self.src = src
            self.i = 0
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            if self.i >= len(frames):
                return False, None
            frame = frames[self.i]
            self.i += 1
            return True, frame.copy()

        def get(self, prop):
            return 30.0

        def release(self):
            self.released = True

    tmp_out = tempfile.mkdtemp(prefix="ieda-smoke-")
    orig_capture = pose_demo.cv2.VideoCapture
    orig_out = pose_demo.OUT_DIR
    orig_model = pose_demo.ensure_model
    try:
        pose_demo.cv2.VideoCapture = FakeCapture
        pose_demo.OUT_DIR = tmp_out
        # use the already-downloaded lite model if present (offline-friendly)
        pose_demo.ensure_model = lambda variant="lite": orig_model("lite")
        args = types.SimpleNamespace(
            camera=0, video=None, webcam=True, model="lite",
            known=None, known_side=None, label=False,
            no_show=True, save_every=2, max_frames=5,
        )
        pose_demo.run_stream(args)   # must not raise
        created = os.listdir(tmp_out)
        assert any(f.endswith("_pose.jpg") for f in created), created
        assert any(f.endswith("_measures.csv") for f in created), created
        assert any(f.endswith("_angles.csv") for f in created), created
    finally:
        pose_demo.cv2.VideoCapture = orig_capture
        pose_demo.OUT_DIR = orig_out
        pose_demo.ensure_model = orig_model
        shutil.rmtree(tmp_out, ignore_errors=True)


if __name__ == "__main__":
    test_posture_status_rules()
    print("PASS posture_status rules (color-coded HUD decisions)")
    test_live_branch_on_fake_camera()
    print("PASS live --webcam code path on a simulated camera (no hardware needed)")
