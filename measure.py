"""
measure.py - geometry helpers that map MediaPipe's 33 pose landmarks to the
Pheasant (1986) anthropometric dimensions used in IEDA 3130 Lab 2.

Nothing here imports mediapipe, so students can unit-test it with plain numpy.

Landmark index reference (BlazePose 33):
 0 nose | 2 L eye | 5 R eye | 7 L ear | 8 R ear
 11 L shoulder | 12 R shoulder | 13 L elbow | 14 R elbow | 15 L wrist | 16 R wrist
 17 L pinky | 18 R pinky | 19 L index | 20 R index | 21 L thumb | 22 R thumb
 23 L hip | 24 R hip | 25 L knee | 26 R knee | 27 L ankle | 28 R ankle
 29 L heel | 30 R heel | 31 L foot index | 32 R foot index
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
    "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
]

POSE_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 2), (2, 7), (0, 5), (5, 8),                      # head
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),    # arms
    (15, 17), (15, 19), (15, 21), (17, 19),              # left hand
    (16, 18), (16, 20), (16, 22), (18, 20),              # right hand
    (11, 23), (12, 24), (23, 24),                        # trunk
    (23, 25), (25, 27), (24, 26), (26, 28),              # legs
    (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32),  # feet
]

L, R = "L", "R"
SIDE_IDX = {
    L: dict(eye=2, ear=7, shoulder=11, elbow=13, wrist=15, index=19, hip=23, knee=25, ankle=27, heel=29),
    R: dict(eye=5, ear=8, shoulder=12, elbow=14, wrist=16, index=20, hip=24, knee=26, ankle=28, heel=30),
}

# (key, display name, Pheasant no., point A, point B, kind, note)
# kind: "dist" = Euclidean, "dy" = vertical offset (A above B is positive)
MEASURE_SPEC = [
    ("shoulder_elbow", "Shoulder-elbow length", 22, "shoulder", "elbow", "dist",
     "acromion->elbow; close to Pheasant 22"),
    ("elbow_fingertip", "Elbow-fingertip length", 23, "elbow", "index", "dist",
     "elbow->index tip; approx. Pheasant 23"),
    ("shoulder_wrist", "Shoulder-wrist length (proxy 25)", 25, "shoulder", "wrist", "dist",
     "Shoulder-grip length ~ shoulder-wrist + ~0.5 hand length"),
    ("hip_knee", "Hip-knee length (proxy 13/14)", 13, "hip", "knee", "dist",
     "joint-centre to joint-centre; UNDER-estimates buttock-knee"),
    ("knee_ankle", "Knee-ankle length (proxy 16)", 16, "knee", "ankle", "dist",
     "approx. lower-leg length; popliteal ht = this + ankle-floor (~7 cm) + shoe 2 cm"),
    ("sit_shoulder_ht", "Shoulder above hip (vertical)", 10, "shoulder", "hip", "dy",
     "side view, seated: proxy for sitting shoulder height"),
    ("sit_eye_ht", "Eye above hip (vertical)", 9, "eye", "hip", "dy",
     "side view, seated: proxy for sitting eye height"),
    ("sit_elbow_ht", "Elbow above hip (vertical)", 11, "elbow", "hip", "dy",
     "side view, seated, upper arm vertical: proxy for sitting elbow height"),
]

# Only straight-line ("dist") measures have a tape-measurable counterpart, so
# only they can be used to calibrate cm/px.  Exported for the CLI help text.
CALIBRATION_KEYS = [row[0] for row in MEASURE_SPEC if row[5] == "dist"]


# --------------------------------------------------------------------------- #
def landmarks_to_arrays(result, width: int, height: int):
    """Convert a PoseLandmarkerResult (first pose) to numpy arrays.

    Returns
    -------
    pts_px : (33, 2) pixel coordinates
    vis    : (33,)  visibility 0..1
    world  : (33, 3) metric coordinates in metres (hip-centre origin) or None
    """
    lms = result.pose_landmarks[0]
    pts_px = np.array([[lm.x * width, lm.y * height] for lm in lms], dtype=float)
    vis = np.array([getattr(lm, "visibility", 1.0) or 0.0 for lm in lms], dtype=float)
    world = None
    if getattr(result, "pose_world_landmarks", None):
        wl = result.pose_world_landmarks[0]
        world = np.array([[lm.x, lm.y, lm.z] for lm in wl], dtype=float)
    return pts_px, vis, world


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def angle_deg(a, b, c) -> Optional[float]:
    """Angle ABC (at vertex b) in degrees, 2-D."""
    a, b, c = (np.asarray(p, dtype=float) for p in (a, b, c))
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return None
    cosang = float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))
    return math.degrees(math.acos(cosang))


def choose_side(vis: np.ndarray) -> str:
    """Pick the body side that is more visible (useful for side-view photos)."""
    l = vis[[11, 13, 15, 23, 25, 27]].mean()
    r = vis[[12, 14, 16, 24, 26, 28]].mean()
    return L if l >= r else R


def compute_measures(pts_px: np.ndarray, vis: np.ndarray, world: Optional[np.ndarray],
                     known: Optional[Dict[str, float]] = None,
                     known_side: Optional[str] = None,
                     vis_thr: float = 0.5) -> List[dict]:
    """Return a list of dict rows for both sides.

    known      : {"shoulder_elbow": 33.5, ...} tape-measured cm values. The mean
                 cm/px ratio from all provided keys calibrates every pixel length.
    known_side : "L"/"R" when the tape values belong to ONE side only.  Without
                 it, every visible side contributes a scale sample (fine for a
                 side-view photo where the far side is hidden; on a front-view
                 photo pass the side you actually measured).
    """
    known = known or {}
    rows = []
    scale_samples = []

    def side_rows(side):
        idx = SIDE_IDX[side]
        out = []
        for key, name, pno, a, b, kind, note in MEASURE_SPEC:
            ia, ib = idx[a], idx[b]
            ok = vis[ia] >= vis_thr and vis[ib] >= vis_thr
            if kind == "dist":
                px = _dist(pts_px[ia], pts_px[ib])
                wcm = _dist(world[ia], world[ib]) * 100 if world is not None else None
            else:  # vertical offset, image y grows downward
                px = float(pts_px[ib, 1] - pts_px[ia, 1])
                wcm = float(world[ib, 1] - world[ia, 1]) * 100 if world is not None else None
            out.append(dict(key=key, measure=name, pheasant_no=pno, side=side, pixels=px,
                            world_cm=wcm, calibrated_cm=None,
                            note=note if ok else "LOW VISIBILITY - " + note))
            if (key in known and kind == "dist" and ok and px > 1
                    and (known_side is None or side == known_side)):
                scale_samples.append(known[key] / px)
        return out

    rows = side_rows(L) + side_rows(R)
    if scale_samples:
        cm_per_px = float(np.mean(scale_samples))
        for r in rows:
            r["calibrated_cm"] = r["pixels"] * cm_per_px
    return rows


def compute_angles(pts_px: np.ndarray, vis: np.ndarray, vis_thr: float = 0.5) -> Dict[str, Optional[float]]:
    """Posture angles relevant to workstation design (BS 7310: elbow ~ 90 deg)."""
    def ok(*ids):
        return all(vis[i] >= vis_thr for i in ids)

    out: Dict[str, Optional[float]] = {}
    for side in (L, R):
        s = SIDE_IDX[side]
        out[f"elbow_flexion_{side}"] = angle_deg(pts_px[s["shoulder"]], pts_px[s["elbow"]], pts_px[s["wrist"]]) \
            if ok(s["shoulder"], s["elbow"], s["wrist"]) else None
        out[f"knee_flexion_{side}"] = angle_deg(pts_px[s["hip"]], pts_px[s["knee"]], pts_px[s["ankle"]]) \
            if ok(s["hip"], s["knee"], s["ankle"]) else None
        out[f"hip_flexion_{side}"] = angle_deg(pts_px[s["shoulder"]], pts_px[s["hip"]], pts_px[s["knee"]]) \
            if ok(s["shoulder"], s["hip"], s["knee"]) else None
    # trunk lean: angle between hip->shoulder (mid-points) and the vertical
    if ok(11, 12, 23, 24):
        sh = (pts_px[11] + pts_px[12]) / 2
        hp = (pts_px[23] + pts_px[24]) / 2
        v = sh - hp
        out["trunk_lean_deg"] = abs(math.degrees(math.atan2(v[0], -v[1])))
    else:
        out["trunk_lean_deg"] = None
    # neck / viewing line: ear-shoulder vs vertical
    side = choose_side(vis)
    s = SIDE_IDX[side]
    if ok(s["ear"], s["shoulder"]):
        v = pts_px[s["ear"]] - pts_px[s["shoulder"]]
        out["neck_forward_deg"] = abs(math.degrees(math.atan2(v[0], -v[1])))
    else:
        out["neck_forward_deg"] = None
    return out
