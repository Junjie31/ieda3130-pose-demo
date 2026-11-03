#!/usr/bin/env python3
"""Unit tests for measure.py - pure geometry, no camera and no mediapipe needed.

Run with either:
    python tests/test_measure.py
    python -m unittest discover -s tests -v
"""
import math
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from measure import (  # noqa: E402
    CALIBRATION_KEYS,
    angle_deg,
    choose_side,
    compute_angles,
    compute_measures,
    landmarks_to_arrays,
)


def make_fixture():
    """Return (pts_px (33,2), vis (33,), world (33,3)) for a standing figure.

    Geometry (image pixels, y grows downwards):
      shoulders y=200, elbows y=300, wrists y=400, index y=400 (straight arm)
      hips y=500, knees y=600, ankles y=700
      L side x=300, R side x=500  -> every "dist" row is 100 px
    World coordinates are metres = px/100, so a 100 px segment is 100 cm.
    """
    pts = np.zeros((33, 2), dtype=float)
    pts[2] = (300, 100)    # left eye
    pts[5] = (500, 100)    # right eye
    pts[7] = (280, 110)    # left ear
    pts[8] = (520, 110)    # right ear
    pts[11], pts[12] = (300, 200), (500, 200)   # shoulders
    pts[13], pts[14] = (300, 300), (500, 300)   # elbows
    pts[15], pts[16] = (300, 400), (500, 400)   # wrists
    pts[19], pts[20] = (300, 400), (500, 400)   # index fingers
    pts[23], pts[24] = (320, 500), (480, 500)   # hips
    pts[25], pts[26] = (320, 600), (480, 600)   # knees
    pts[27], pts[28] = (320, 700), (480, 700)   # ankles
    pts[29], pts[30] = (320, 720), (480, 720)   # heels
    pts[31], pts[32] = (350, 720), (450, 720)   # foot index
    vis = np.ones(33, dtype=float)
    world = np.zeros((33, 3), dtype=float)
    world[:, 0] = pts[:, 0] / 100.0
    world[:, 1] = pts[:, 1] / 100.0
    return pts, vis, world


class TestAngle(unittest.TestCase):
    def test_right_angle(self):
        self.assertAlmostEqual(angle_deg((0, 1), (0, 0), (1, 0)), 90.0, places=6)

    def test_straight_line(self):
        self.assertAlmostEqual(angle_deg((0, -1), (0, 0), (0, 1)), 180.0, places=6)

    def test_45_degrees(self):
        self.assertAlmostEqual(angle_deg((1, 0), (0, 0), (1, 1)), 45.0, places=6)

    def test_degenerate_returns_none(self):
        self.assertIsNone(angle_deg((0, 0), (0, 0), (1, 1)))


class TestChooseSide(unittest.TestCase):
    def test_more_visible_side_wins(self):
        pts, vis, _ = make_fixture()
        vis[[12, 14, 16, 24, 26, 28]] = 0.1
        self.assertEqual(choose_side(vis), "L")
        vis[[11, 13, 15, 23, 25, 27]] = 0.0
        vis[[12, 14, 16, 24, 26, 28]] = 1.0
        self.assertEqual(choose_side(vis), "R")


class TestComputeMeasures(unittest.TestCase):
    def setUp(self):
        self.pts, self.vis, self.world = make_fixture()

    def test_rows_are_both_sides_of_every_measure(self):
        rows = compute_measures(self.pts, self.vis, self.world)
        self.assertEqual(len(rows), 16)  # 8 measures x 2 sides

    def test_pixel_lengths(self):
        rows = {(r["key"], r["side"]): r for r in
                compute_measures(self.pts, self.vis, self.world)}
        expected = {"shoulder_elbow": 100.0, "elbow_fingertip": 100.0,
                    "shoulder_wrist": 200.0, "hip_knee": 100.0, "knee_ankle": 100.0}
        for key, px in expected.items():
            for side in ("L", "R"):
                self.assertAlmostEqual(rows[(key, side)]["pixels"], px, places=6)

    def test_vertical_rows_are_positive_when_a_is_above_b(self):
        rows = {(r["key"], r["side"]): r for r in
                compute_measures(self.pts, self.vis, self.world)}
        self.assertAlmostEqual(rows[("sit_shoulder_ht", "L")]["pixels"], 300.0)
        self.assertAlmostEqual(rows[("sit_eye_ht", "L")]["pixels"], 400.0)
        self.assertAlmostEqual(rows[("sit_elbow_ht", "L")]["pixels"], 200.0)

    def test_world_cm_equals_known_geometry(self):
        rows = {(r["key"], r["side"]): r for r in
                compute_measures(self.pts, self.vis, self.world)}
        self.assertAlmostEqual(rows[("shoulder_elbow", "L")]["world_cm"], 100.0, places=6)

    def test_no_world_means_no_world_values(self):
        rows = compute_measures(self.pts, self.vis, None, known={"shoulder_elbow": 33.5})
        for r in rows:
            self.assertIsNone(r["world_cm"])
        # ... but pixel calibration still works without world landmarks
        rows = {(r["key"], r["side"]): r for r in
                compute_measures(self.pts, self.vis, None, known={"shoulder_elbow": 33.5})}
        self.assertAlmostEqual(rows[("shoulder_elbow", "L")]["calibrated_cm"], 33.5, places=6)

    def test_low_visibility_is_flagged(self):
        vis = self.vis.copy()
        vis[13] = 0.0  # left elbow
        rows = {(r["key"], r["side"]): r for r in
                compute_measures(self.pts, vis, self.world)}
        self.assertIn("LOW VISIBILITY", rows[("shoulder_elbow", "L")]["note"])
        self.assertNotIn("LOW VISIBILITY", rows[("shoulder_elbow", "R")]["note"])

    def test_calibration_round_trip_on_symmetric_body(self):
        rows = {(r["key"], r["side"]): r for r in
                compute_measures(self.pts, self.vis, self.world,
                                 known={"shoulder_elbow": 33.5})}
        self.assertAlmostEqual(rows[("shoulder_elbow", "L")]["calibrated_cm"], 33.5, places=6)
        self.assertAlmostEqual(rows[("shoulder_elbow", "R")]["calibrated_cm"], 33.5, places=6)
        # the same cm/px scale is applied to the other segments
        self.assertAlmostEqual(rows[("knee_ankle", "L")]["calibrated_cm"], 33.5, places=6)

    def test_known_side_restricts_the_calibration_sample(self):
        pts = self.pts.copy()
        pts[12] = (500, 250)  # right shoulder -> right shoulder-elbow = 50 px, left = 100 px
        pts[14] = (500, 300)
        rows = {(r["key"], r["side"]): r for r in
                compute_measures(pts, self.vis, self.world,
                                 known={"shoulder_elbow": 33.5}, known_side="L")}
        self.assertAlmostEqual(rows[("shoulder_elbow", "L")]["calibrated_cm"], 33.5, places=6)
        self.assertAlmostEqual(rows[("shoulder_elbow", "R")]["calibrated_cm"], 16.75, places=6)

    def test_without_known_side_both_sides_are_averaged(self):
        pts = self.pts.copy()
        pts[12] = (500, 250)
        pts[14] = (500, 300)
        rows = {(r["key"], r["side"]): r for r in
                compute_measures(pts, self.vis, self.world,
                                 known={"shoulder_elbow": 33.5})}
        # mean of (33.5/100, 33.5/50) = 0.5025 cm/px
        self.assertAlmostEqual(rows[("shoulder_elbow", "L")]["calibrated_cm"], 50.25, places=6)
        self.assertAlmostEqual(rows[("shoulder_elbow", "R")]["calibrated_cm"], 25.125, places=6)


class TestComputeAngles(unittest.TestCase):
    def setUp(self):
        self.pts, self.vis, _ = make_fixture()

    def test_straight_limbs(self):
        angles = compute_angles(self.pts, self.vis)
        self.assertAlmostEqual(angles["elbow_flexion_L"], 180.0, places=6)
        self.assertAlmostEqual(angles["knee_flexion_R"], 180.0, places=6)
        self.assertAlmostEqual(angles["trunk_lean_deg"], 0.0, places=6)

    def test_ninety_degree_elbow(self):
        pts = self.pts.copy()
        pts[15] = (400, 300)  # forearm horizontal
        angles = compute_angles(pts, self.vis)
        self.assertAlmostEqual(angles["elbow_flexion_L"], 90.0, places=6)

    def test_missing_chain_returns_none(self):
        vis = self.vis.copy()
        vis[15] = 0.0
        angles = compute_angles(self.pts, vis)
        self.assertIsNone(angles["elbow_flexion_L"])


class TestLandmarksToArrays(unittest.TestCase):
    def _fake_lm(self, x, y, z=0.0, visibility=1.0):
        return SimpleNamespace(x=x, y=y, z=z, visibility=visibility)

    def test_normalised_to_pixels_and_world(self):
        lms = [self._fake_lm(0.5, 0.25, 0.0, 0.9) for _ in range(33)]
        result = SimpleNamespace(pose_landmarks=[lms], pose_world_landmarks=None)
        pts, vis, world = landmarks_to_arrays(result, 200, 100)
        self.assertEqual(pts.shape, (33, 2))
        self.assertAlmostEqual(pts[0, 0], 100.0)
        self.assertAlmostEqual(pts[0, 1], 25.0)
        self.assertAlmostEqual(vis[0], 0.9)
        self.assertIsNone(world)

    def test_world_landmarks_in_metres(self):
        lms = [self._fake_lm(0.5, 0.25, 0.0, 1.0) for _ in range(33)]
        wlms = [self._fake_lm(0.1, 0.2, 0.3, 1.0) for _ in range(33)]
        result = SimpleNamespace(pose_landmarks=[lms], pose_world_landmarks=[wlms])
        _, _, world = landmarks_to_arrays(result, 200, 100)
        self.assertEqual(world.shape, (33, 3))
        np.testing.assert_allclose(world[0], [0.1, 0.2, 0.3])


class TestCalibrationKeys(unittest.TestCase):
    def test_keys_are_the_distance_measures(self):
        self.assertEqual(
            CALIBRATION_KEYS,
            ["shoulder_elbow", "elbow_fingertip", "shoulder_wrist", "hip_knee", "knee_ankle"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
