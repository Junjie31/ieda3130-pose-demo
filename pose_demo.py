#!/usr/bin/env python3
"""
IEDA 3130 Ergonomics and Safety Management - Pose Estimation Demo
=================================================================
An "OpenPose-style" skeleton demo built on MediaPipe Pose Landmarker (33 body
landmarks). It links the keypoints to the Pheasant anthropometric dimensions
used in Lab 2 (Anthropometry: Workstation Design) and to BS 7310 posture
criteria (elbow ~90 deg at keyboard height).

Usage
-----
  python pose_demo.py --image photo.jpg                 # single photo
  python pose_demo.py --image photo.jpg --known shoulder_elbow=33.5
  python pose_demo.py --webcam                          # live camera (press s = save, q = quit)
  python pose_demo.py --video clip.mp4 --no-show        # video file, headless

Outputs are written to ./outputs/<name>_pose.jpg and ./outputs/<name>_measures.csv

TA: Junjie HOU (IEDA 3130, HKUST)
"""
import argparse
import csv
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError as e:  # pragma: no cover
    sys.exit(
        "mediapipe is not installed. Run:  python -m pip install -r requirements.txt\n"
        f"Original error: {e}"
    )

from measure import (
    LANDMARK_NAMES,
    POSE_CONNECTIONS,
    compute_angles,
    compute_measures,
    landmarks_to_arrays,
)

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
OUT_DIR = os.path.join(os.getcwd(), "outputs")


# --------------------------------------------------------------------------- #
# Model handling
# --------------------------------------------------------------------------- #
def ensure_model(variant: str = "lite") -> str:
    """Download the .task model bundle once and return its local path."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"pose_landmarker_{variant}.task")
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    url = MODEL_URLS[variant]
    print(f"[info] downloading model ({variant}) ...\n       {url}")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        sys.exit(
            f"[error] model download failed: {e}\n"
            f"        Download it manually in a browser and save as:\n        {path}"
        )
    print(f"[info] saved -> {path}")
    return path


def build_landmarker(model_path: str, mode: str):
    """mode: 'image' or 'video'."""
    running = (
        mp_vision.RunningMode.IMAGE if mode == "image" else mp_vision.RunningMode.VIDEO
    )
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=running,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def to_mp_image(frame_bgr: np.ndarray):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


# --------------------------------------------------------------------------- #
# Drawing (pure OpenCV - independent of mediapipe drawing utils versions)
# --------------------------------------------------------------------------- #
def draw_skeleton(frame: np.ndarray, pts_px: np.ndarray, vis: np.ndarray,
                  vis_thr: float = 0.5, label: bool = False) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    thick = max(1, int(round(min(h, w) / 300)))
    for a, b in POSE_CONNECTIONS:
        if vis[a] < vis_thr or vis[b] < vis_thr:
            continue
        pa, pb = tuple(int(v) for v in pts_px[a]), tuple(int(v) for v in pts_px[b])
        cv2.line(out, pa, pb, (0, 255, 0), thick, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts_px):
        if vis[i] < vis_thr:
            continue
        cv2.circle(out, (int(x), int(y)), thick * 2 + 1, (0, 0, 255), -1, cv2.LINE_AA)
        if label:
            cv2.putText(out, str(i), (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35 * thick, (255, 255, 0), 1, cv2.LINE_AA)
    return out


def put_panel(frame: np.ndarray, lines, origin=(10, 25)) -> np.ndarray:
    x, y = origin
    for text in lines:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return frame


# --------------------------------------------------------------------------- #
# Core processing of one frame
# --------------------------------------------------------------------------- #
def process_result(frame_bgr, result, known=None, label=False):
    """Return (annotated_frame, measures_rows, angles_dict) or (frame, [], {})."""
    if not result.pose_landmarks:
        return put_panel(frame_bgr.copy(), ["No person detected"]), [], {}
    h, w = frame_bgr.shape[:2]
    pts_px, vis, world_m = landmarks_to_arrays(result, w, h)
    annotated = draw_skeleton(frame_bgr, pts_px, vis, label=label)
    rows = compute_measures(pts_px, vis, world_m, known=known)
    angles = compute_angles(pts_px, vis)
    return annotated, rows, angles


def save_outputs(stem: str, annotated, rows, angles):
    os.makedirs(OUT_DIR, exist_ok=True)
    img_path = os.path.join(OUT_DIR, f"{stem}_pose.jpg")
    csv_path = os.path.join(OUT_DIR, f"{stem}_measures.csv")
    cv2.imwrite(img_path, annotated)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow(["measure", "pheasant_no", "side", "pixels", "world_cm", "calibrated_cm", "note"])
        for r in rows:
            wri.writerow([r["measure"], r["pheasant_no"], r["side"],
                          f'{r["pixels"]:.1f}',
                          "" if r["world_cm"] is None else f'{r["world_cm"]:.1f}',
                          "" if r["calibrated_cm"] is None else f'{r["calibrated_cm"]:.1f}',
                          r["note"]])
        wri.writerow([])
        wri.writerow(["angle_deg", "", "", "", "", "", ""])
        for k, v in angles.items():
            wri.writerow([k, "", "", "", "", f"{v:.1f}" if v is not None else "", ""])
    print(f"[ok] image -> {img_path}\n[ok] csv   -> {csv_path}")


def print_table(rows, angles):
    print("\n{:<34} {:>6} {:>8} {:>10} {:>12}".format("measure", "No.", "px", "world_cm", "calib_cm"))
    print("-" * 76)
    for r in rows:
        print("{:<34} {:>6} {:>8.1f} {:>10} {:>12}".format(
            f'{r["measure"]} ({r["side"]})', r["pheasant_no"], r["pixels"],
            "-" if r["world_cm"] is None else f'{r["world_cm"]:.1f}',
            "-" if r["calibrated_cm"] is None else f'{r["calibrated_cm"]:.1f}'))
    print("\nJoint angles (deg):")
    for k, v in angles.items():
        print(f"  {k:<28} {'-' if v is None else f'{v:.1f}'}")


def parse_known(items):
    """--known shoulder_elbow=33.5 --known elbow_fingertip=44 -> dict (cm)."""
    known = {}
    for it in items or []:
        if "=" not in it:
            sys.exit(f"--known expects name=cm, got {it!r}")
        k, v = it.split("=", 1)
        known[k.strip()] = float(v)
    return known


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_image(args):
    frame = cv2.imread(args.image)
    if frame is None:
        sys.exit(f"[error] cannot read image: {args.image}")
    if max(frame.shape[:2]) > 1600:  # keep display manageable
        s = 1600 / max(frame.shape[:2])
        frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    model = ensure_model(args.model)
    with build_landmarker(model, "image") as lm:
        result = lm.detect(to_mp_image(frame))
    annotated, rows, angles = process_result(frame, result, known=parse_known(args.known), label=args.label)
    if rows:
        print_table(rows, angles)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    save_outputs(stem, annotated, rows, angles)
    if not args.no_show:
        cv2.imshow("IEDA3130 pose demo - press any key", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_stream(args):
    src = args.camera if args.video is None else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"[error] cannot open source {src!r}. Try --camera 1 (or 0) / check camera permission.")
    model = ensure_model(args.model)
    known = parse_known(args.known)
    fps_hist, t_prev, ts_ms, n_saved = [], time.time(), 0, 0
    with build_landmarker(model, "video") as lm:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[info] end of stream")
                break
            if args.video is None:
                frame = cv2.flip(frame, 1)  # mirror webcam
            ts_ms += 33 if args.video is None else int(1000 / max(cap.get(cv2.CAP_PROP_FPS) or 30, 1))
            result = lm.detect_for_video(to_mp_image(frame), ts_ms)
            annotated, rows, angles = process_result(frame, result, known=known, label=args.label)
            now = time.time()
            fps_hist.append(1.0 / max(now - t_prev, 1e-6)); t_prev = now
            fps = np.mean(fps_hist[-20:])
            lines = [f"FPS {fps:4.1f}   model={args.model}   [s]=save  [q]=quit"]
            for k in ("elbow_flexion_L", "elbow_flexion_R", "knee_flexion_L", "knee_flexion_R", "trunk_lean_deg"):
                if k in angles and angles[k] is not None:
                    lines.append(f"{k:<16} {angles[k]:6.1f} deg")
            put_panel(annotated, lines)
            if not args.no_show:
                cv2.imshow("IEDA3130 pose demo", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                if key == ord("s") and rows:
                    n_saved += 1
                    save_outputs(f"frame_{n_saved:03d}", annotated, rows, angles)
                    print_table(rows, angles)
    cap.release()
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="IEDA3130 pose estimation demo (MediaPipe Pose Landmarker)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="path to a photo (jpg/png)")
    g.add_argument("--video", help="path to a video file")
    g.add_argument("--webcam", action="store_true", help="use the laptop camera")
    p.add_argument("--camera", type=int, default=0, help="camera index for --webcam (default 0)")
    p.add_argument("--model", choices=list(MODEL_URLS), default="lite", help="lite (fast) / full / heavy (accurate)")
    p.add_argument("--known", action="append", metavar="NAME=CM",
                   help="tape-measured length in cm for calibration, e.g. --known shoulder_elbow=33.5")
    p.add_argument("--label", action="store_true", help="draw landmark indices (0-32)")
    p.add_argument("--no-show", action="store_true", help="do not open a window (headless / Colab)")
    args = p.parse_args()
    if args.image:
        run_image(args)
    else:
        run_stream(args)


if __name__ == "__main__":
    main()
