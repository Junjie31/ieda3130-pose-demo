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
  python pose_demo.py --image photo.jpg --known shoulder_elbow=33.5 --known-side R
  python pose_demo.py --webcam                          # live camera (press s = save, q = quit)
  python pose_demo.py --video clip.mp4 --no-show --save-every 15   # headless video

Outputs are written NEXT TO the script (not next to your shell) in
./outputs/<name>_pose.jpg, ./outputs/<name>_measures.csv and
./outputs/<name>_angles.csv.

TA: Junjie HOU (IEDA 3130, HKUST)
"""
import argparse
import contextlib
import csv
import os
import sys
import tempfile
import time
import urllib.request

# --------------------------------------------------------------------------- #
# Quiet MediaPipe / TensorFlow-Lite C++ logging BEFORE those packages are
# imported, otherwise students see scary-looking W0000 / inference_feedback_manager
# lines on every run.  (Must happen before "import mediapipe".)
# --------------------------------------------------------------------------- #
os.environ.setdefault("GLOG_minloglevel", "2")      # 0=INFO 1=WARN 2=ERROR
os.environ.setdefault("GLOG_logtostderr", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silence TFLite delegate INFO
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

# Make console output safe for non-ASCII paths (Chinese/Japanese folder names are
# common in students' Downloads folders).  Without this, printing a path that
# contains e.g. Chinese characters can crash with UnicodeEncodeError on Windows.
for _stream in (sys.stdout, sys.stderr):
    try:  # Python >= 3.7
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover - old Python / exotic stream
        pass

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

try:  # keep the C++/absl loggers quiet for the work that happens after import
    import absl.logging

    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:  # pragma: no cover
    pass

from measure import (
    CALIBRATION_KEYS,
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR = os.path.join(BASE_DIR, "outputs")   # always next to the code, not the cwd


# --------------------------------------------------------------------------- #
# Quiet MediaPipe's C++ logging
# --------------------------------------------------------------------------- #
# MediaPipe/TFLite print glog-style lines directly to C-level file descriptor 2
# ("INFO: Created TensorFlow Lite XNNPACK delegate...", "W0000 ...
# inference_feedback_manager.cc", "WARNING: Logging before InitGoogle()...").
# Environment variables are ignored because glog logs before it is initialised,
# so we temporarily point fd 2 at os.devnull around the MediaPipe calls.  Set
# IEDA3130_VERBOSE=1 to keep the raw logs (useful when debugging a failure).
@contextlib.contextmanager
def quiet_mediapipe():
    if os.environ.get("IEDA3130_VERBOSE"):
        yield
        return
    try:
        err_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):  # stderr is not a real fd
        yield
        return
    saved = devnull = None
    try:
        saved = os.dup(err_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, err_fd)
        yield
    finally:
        if saved is not None:
            try:
                os.dup2(saved, err_fd)
            except OSError:
                pass
            os.close(saved)
        if devnull is not None:
            os.close(devnull)


# --------------------------------------------------------------------------- #
# Unicode-safe image I/O
# --------------------------------------------------------------------------- #
# OpenCV's imread/imwrite use the *ANSI* file API on Windows, so they fail for
# paths containing Chinese characters, emoji, etc.  Reading/writing through
# numpy / Python's own file API fixes that and costs nothing.
def imread_unicode(path: str):
    """cv2.imread replacement that also works with non-ASCII file paths."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_unicode(path: str, image) -> bool:
    """cv2.imwrite replacement that also works with non-ASCII file paths."""
    ext = os.path.splitext(path)[1]
    if not ext:
        ext, path = ".jpg", path + ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    try:
        with open(path, "wb") as fh:
            fh.write(buf.tobytes())
    except OSError:
        return False
    return True


def _ascii_temp_dir():
    """Return a writable directory whose absolute path is pure ASCII, or None."""
    candidates = [tempfile.gettempdir(),
                  os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "Temp")]
    for cand in candidates:
        try:
            cand.encode("ascii")
        except UnicodeEncodeError:
            continue
        try:
            os.makedirs(cand, exist_ok=True)
            probe = os.path.join(cand, ".ieda-write-test")
            with open(probe, "w", encoding="ascii") as fh:
                fh.write("ok")
            os.remove(probe)
            return cand
        except OSError:
            continue
    return None


def mediapipe_safe_model_path(path: str) -> str:
    """MediaPipe's C++ loader cannot open non-ASCII paths.

    If the model file lives in e.g. a Chinese-named folder, copy it to a temp
    folder with a pure-ASCII absolute path and use that copy instead.  The copy
    is reused on later runs (same size => no re-copy).
    """
    try:
        path.encode("ascii")
        return path
    except UnicodeEncodeError:
        pass
    tmpdir = _ascii_temp_dir()
    if tmpdir is None:
        print("[warn] the project folder contains non-ASCII characters and no ASCII "
              "temp folder was found.\n"
              "       MediaPipe may fail to load the model. Move the project to a "
              "path like C:\\IEDA3130.")
        return path
    import hashlib
    import shutil
    tag = hashlib.sha1(path.encode("utf-8")).hexdigest()[:10]
    dst = os.path.join(tmpdir, f"ieda3130_model_{tag}.task")
    try:
        if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(path)):
            shutil.copyfile(path, dst)
        print(f"[info] non-ASCII project path detected - using ASCII model copy:\n       {dst}")
    except OSError as e:
        print(f"[warn] could not copy the model to an ASCII path ({e}); trying the "
              f"original path.")
        return path
    return dst


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
    tmp = path + ".part"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ieda3130-pose-demo"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            last_step = -1
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if total:
                    step = int(100.0 * done / total) // 5   # print every ~5 %
                    if step != last_step:
                        last_step = step
                        print(f"\r[info] downloading {100.0 * done / total:5.1f}% "
                              f"({done / 1e6:5.1f} / {total / 1e6:.1f} MB)",
                              end="", flush=True)
                else:
                    print(f"\r[info] downloading {done / 1e6:5.1f} MB", end="", flush=True)
        if total:
            print()
        os.replace(tmp, path)
    except Exception as e:
        try:
            os.remove(tmp)
        except OSError:
            pass
        sys.exit(
            f"[error] model download failed: {e}\n"
            f"        Check your network, or download the file manually in a browser\n"
            f"        and save it as:\n        {path}"
        )
    print(f"[info] saved -> {path}")
    return path


def build_landmarker(model_path: str, mode: str):
    """mode: 'image' or 'video'."""
    model_path = mediapipe_safe_model_path(model_path)
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
    with quiet_mediapipe():
        return mp_vision.PoseLandmarker.create_from_options(options)


def detect_image(lm, frame_bgr):
    """Run one image-mode inference with the C++ log noise suppressed."""
    with quiet_mediapipe():
        return lm.detect(to_mp_image(frame_bgr))


def detect_video(lm, frame_bgr, timestamp_ms: int):
    """Run one video-mode inference with the C++ log noise suppressed."""
    with quiet_mediapipe():
        return lm.detect_for_video(to_mp_image(frame_bgr), timestamp_ms)


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
def process_result(frame_bgr, result, known=None, known_side=None, label=False):
    """Return (annotated_frame, measures_rows, angles_dict) or (frame, [], {})."""
    if not result.pose_landmarks:
        return put_panel(frame_bgr.copy(), ["No person detected"]), [], {}
    h, w = frame_bgr.shape[:2]
    pts_px, vis, world_m = landmarks_to_arrays(result, w, h)
    annotated = draw_skeleton(frame_bgr, pts_px, vis, label=label)
    rows = compute_measures(pts_px, vis, world_m, known=known, known_side=known_side)
    angles = compute_angles(pts_px, vis)
    return annotated, rows, angles


def save_outputs(stem: str, annotated, rows, angles):
    os.makedirs(OUT_DIR, exist_ok=True)
    img_path = os.path.join(OUT_DIR, f"{stem}_pose.jpg")
    csv_path = os.path.join(OUT_DIR, f"{stem}_measures.csv")
    ang_path = os.path.join(OUT_DIR, f"{stem}_angles.csv")
    wrote_img = imwrite_unicode(img_path, annotated)
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wri = csv.writer(f)
            wri.writerow(["measure", "pheasant_no", "side", "pixels", "world_cm", "calibrated_cm", "note"])
            for r in rows:
                wri.writerow([r["measure"], r["pheasant_no"], r["side"],
                              f'{r["pixels"]:.1f}',
                              "" if r["world_cm"] is None else f'{r["world_cm"]:.1f}',
                              "" if r["calibrated_cm"] is None else f'{r["calibrated_cm"]:.1f}',
                              r["note"]])
        with open(ang_path, "w", newline="", encoding="utf-8") as f:
            wri = csv.writer(f)
            wri.writerow(["angle", "degrees"])
            for k, v in angles.items():
                wri.writerow([k, "" if v is None else f"{v:.1f}"])
    except OSError as e:
        print(f"[error] could not write CSV: {e}")
        return
    if wrote_img:
        print(f"[ok] image    -> {img_path}")
    else:
        print(f"[error] could not write image (check permissions / disk space): {img_path}")
    print(f"[ok] measures -> {csv_path}")
    print(f"[ok] angles   -> {ang_path}")


def print_table(rows, angles):
    width = max(34, max(len(r["measure"]) for r in rows) + 2) if rows else 34
    print("\n{:<{w}} {:>1} {:>6} {:>9} {:>11} {:>10}".format(
        "measure", "", "No.", "px", "world_cm", "calib_cm", w=width))
    print("-" * (width + 42))
    any_low = False
    for r in rows:
        low = r["note"].startswith("LOW VISIBILITY")
        any_low = any_low or low
        print("{:<{w}} {:>1} {:>6} {:>9.1f} {:>11} {:>10}".format(
            f'{r["measure"]} ({r["side"]})', "!" if low else "", r["pheasant_no"],
            r["pixels"],
            "-" if r["world_cm"] is None else f'{r["world_cm"]:.1f}',
            "-" if r["calibrated_cm"] is None else f'{r["calibrated_cm"]:.1f}', w=width))
    if any_low:
        print("\n[!] Rows marked '!' have at least one landmark with visibility < 0.5.")
        print("    Their pixel/world values are extrapolated - do NOT record them on the worksheet.")
        print("    Re-take the photo with the whole body visible (README section 4).")
    print("\nJoint angles (deg):")
    for k, v in angles.items():
        print(f"  {k:<28} {'-' if v is None else f'{v:.1f}'}")
    if any(v is None for v in angles.values()):
        print("  ('-' = the required landmarks were not visible enough to compute this angle)")


def parse_known(items, known_side=None):
    """--known shoulder_elbow=33.5 --known elbow_fingertip=44 -> dict (cm).

    Valid keys are the *distance* measures in measure.MEASURE_SPEC:
      shoulder_elbow, elbow_fingertip, shoulder_wrist, hip_knee, knee_ankle
    Bad input is reported as a normal one-line error, never as a traceback.
    """
    known = {}
    for it in items or []:
        if "=" not in it:
            sys.exit(f"[error] --known expects NAME=CM (e.g. shoulder_elbow=33.5), got {it!r}")
        k, v = it.split("=", 1)
        k = k.strip().lower()
        try:
            value = float(v)
        except ValueError:
            sys.exit(f"[error] --known {k}: '{v.strip()}' is not a number "
                     f"(expected centimetres, e.g. {k}=33.5)")
        if not np.isfinite(value) or value <= 0:
            sys.exit(f"[error] --known {k}: the tape value must be a positive number of "
                     f"centimetres, got {value}")
        if value > 200:
            sys.exit(f"[error] --known {k}: {value} cm is not a plausible body-segment length "
                     f"(tape values are typically 25-65 cm)")
        if k not in CALIBRATION_KEYS:
            sys.exit(
                f"[error] unknown --known key: {k!r}\n"
                f"        valid keys are: {', '.join(CALIBRATION_KEYS)}\n"
                f"        (only straight-line lengths can be used for calibration)"
            )
        known[k] = value
    return known


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_image(args):
    known = parse_known(args.known, args.known_side)   # fail fast, before model load
    frame = imread_unicode(args.image)
    if frame is None:
        sys.exit(f"[error] cannot read image: {args.image}\n"
                 f"        Check the path (spaces are fine, quotes help), the file extension "
                 f"(.jpg/.png) and that the file really is a photo.")
    if max(frame.shape[:2]) > 1600:  # keep display manageable
        s = 1600 / max(frame.shape[:2])
        frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    model = ensure_model(args.model)
    with build_landmarker(model, "image") as lm:
        result = detect_image(lm, frame)
    annotated, rows, angles = process_result(
        frame, result, known=known, known_side=args.known_side, label=args.label)
    if rows:
        print_table(rows, angles)
    else:
        print("[warn] no person detected in this photo.\n"
              "       Make sure the WHOLE body is in frame, the lighting is decent and\n"
              "       nothing (bag, chair, loose clothing) hides the joints.  Results were\n"
              "       still saved with the 'No person detected' label.")
    stem = os.path.splitext(os.path.basename(args.image))[0]
    save_outputs(stem, annotated, rows, angles)
    if not args.no_show:
        try:
            cv2.imshow("IEDA3130 pose demo - press any key", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:  # headless Linux / WSL / remote SSH
            cv2.destroyAllWindows()
            print(f"[warn] cannot open a window here ({e}); the annotated image was still saved "
                  f"in outputs/.  Use --no-show next time.")


def run_stream(args):
    known = parse_known(args.known, args.known_side)   # fail fast, before opening the source
    src = args.camera if args.video is None else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        if args.video is None:
            sys.exit(f"[error] cannot open camera {args.camera}.\n"
                     f"        Try --camera 0 and --camera 1, close any app that uses the camera\n"
                     f"        (Zoom, Teams, Camera), and check the camera permission in\n"
                     f"        Windows Settings > Privacy > Camera / macOS System Settings > Privacy.")
        sys.exit(f"[error] cannot open video file: {args.video}\n"
                 f"        Check the path and that it is a real video (mp4/avi/mov).")
    if args.video is not None:
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if 0 < n_frames <= 1:
            print("[warn] this file seems to contain only one frame - did you mean --image?")
        if args.no_show and not args.save_every:
            print("[hint] headless video mode does not save anything by itself;\n"
                  "       add --save-every 15 to write annotated frames + CSVs into outputs/.")
    if args.webcam and args.no_show:
        print("[warn] --webcam with --no-show runs without a window or saved output.\n"
              "       Add --save-every 15 to save frames, or press Ctrl+C to stop.")
    stem = (os.path.splitext(os.path.basename(args.video))[0]
            if args.video is not None else "webcam")
    model = ensure_model(args.model)
    fps_hist, t_prev, ts_ms, n_saved, frame_idx = [], time.time(), 0, 0, 0
    win = "IEDA3130 pose demo"
    with build_landmarker(model, "video") as lm:
        while True:
            if args.max_frames and frame_idx >= args.max_frames:
                print(f"[info] stopping after --max-frames {args.max_frames}")
                break
            ok, frame = cap.read()
            if not ok:
                print("[info] end of stream")
                break
            frame_idx += 1
            if args.video is None:
                frame = cv2.flip(frame, 1)  # mirror webcam
            ts_ms += 33 if args.video is None else int(1000 / max(cap.get(cv2.CAP_PROP_FPS) or 30, 1))
            result = detect_video(lm, frame, ts_ms)
            annotated, rows, angles = process_result(
                frame, result, known=known, known_side=args.known_side, label=args.label)
            if args.save_every and frame_idx % args.save_every == 0 and rows:
                n_saved += 1
                save_outputs(f"{stem}_frame_{frame_idx:04d}", annotated, rows, angles)
            now = time.time()
            fps_hist.append(1.0 / max(now - t_prev, 1e-6)); t_prev = now
            fps = np.mean(fps_hist[-20:])
            lines = [f"FPS {fps:4.1f}   model={args.model}   [s]=save  [q]=quit"]
            for k in ("elbow_flexion_L", "elbow_flexion_R", "knee_flexion_L", "knee_flexion_R", "trunk_lean_deg"):
                if k in angles and angles[k] is not None:
                    lines.append(f"{k:<16} {angles[k]:6.1f} deg")
            put_panel(annotated, lines)
            if not args.no_show:
                try:
                    cv2.imshow(win, annotated)
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error as e:
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(f"[error] cannot open a window here ({e}).\n"
                             f"        Use --video clip.mp4 --no-show --save-every 15 on headless machines.")
                if key == ord("q") or key == 27:
                    break
                if key == ord("s") and rows:
                    n_saved += 1
                    save_outputs(f"frame_{n_saved:03d}", annotated, rows, angles)
                    print_table(rows, angles)
                try:  # closing the window with the X button should also stop the camera
                    if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
    cap.release()
    cv2.destroyAllWindows()
    if args.save_every and n_saved == 0:
        print("[info] --save-every matched no frame with a detected person "
              "(try a larger --save-every or a clearer video).")


def main():
    p = argparse.ArgumentParser(
        description="IEDA3130 pose estimation demo (MediaPipe Pose Landmarker)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="path to a photo (jpg/png)")
    g.add_argument("--video", help="path to a video file")
    g.add_argument("--webcam", action="store_true", help="use the laptop camera")
    p.add_argument("--camera", type=int, default=0, help="camera index for --webcam (default 0)")
    p.add_argument("--model", choices=list(MODEL_URLS), default="lite",
                   help="lite (fast) / full / heavy (accurate)")
    p.add_argument("--known", action="append", metavar="NAME=CM",
                   help="tape-measured length in cm for calibration, e.g. --known shoulder_elbow=33.5 "
                        "(valid names: " + ", ".join(CALIBRATION_KEYS) + ")")
    p.add_argument("--known-side", choices=["L", "R"], default=None,
                   help="which body side the --known tape values refer to (default: all visible sides)")
    p.add_argument("--label", action="store_true", help="draw landmark indices (0-32)")
    p.add_argument("--no-show", action="store_true", help="do not open a window (headless / Colab)")
    p.add_argument("--save-every", type=int, default=0, metavar="N",
                   help="video/webcam only: save the annotated frame and CSVs every Nth frame "
                        "to outputs/ (0 = off; useful on headless machines)")
    p.add_argument("--max-frames", type=int, default=0, metavar="N",
                   help="video/webcam only: stop after N frames (0 = no limit)")
    args = p.parse_args()
    if args.image:
        run_image(args)
    else:
        run_stream(args)


if __name__ == "__main__":
    main()
