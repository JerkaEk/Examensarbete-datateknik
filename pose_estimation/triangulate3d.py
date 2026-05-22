"""
triangulate3d.py
================

Convert paired 2-D landmarks into 3-D coordinates via linear
triangulation and calibrated camera parameters.

Example
-------
$ python triangulate3d.py --pose pose2d.json --out pose3d.json
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2 as cv
import logging

log = logging.getLogger(__name__)

from utils.utils_io import load_json, save_json, load_intrinsics, load_extrinsics


class Triangulator:
    def __init__(self, k0: np.ndarray, k1: np.ndarray, dist0: np.ndarray,
                 dist1: np.ndarray, r1: np.ndarray, t1: np.ndarray):
        # camera0 at origin, no rotation/translation
        r0, t0 = np.eye(3), np.zeros((3, 1))
        self.k0 = k0
        self.k1 = k1
        self.dist0 = dist0
        self.dist1 = dist1
        self.p0 = self.build_projection(k0, r0, t0)
        self.p1 = self.build_projection(k1, r1, t1)

    def build_projection(self, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Return 3x4 projection matrix P = K · [R | t]."""
        Rt = np.hstack((R, t))
        return K @ Rt

    def undistort(self, pts: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
        """Undistort a Nx2 array, preserving NaN rows."""
        if dist is None:
            return pts
        out = pts.copy().astype(np.float32)
        valid = np.isfinite(out).all(axis=1)
        if not valid.any():
            return out
        valid_pts = out[valid].reshape(-1, 1, 2)
        undist = cv.undistortPoints(valid_pts, K, dist, P=K).reshape(-1, 2)
        out[valid] = undist
        return out

    def triangulate(self, pts0: np.ndarray, pts1: np.ndarray) -> np.ndarray:
        """Triangulate pairs of points (Nx2, Nx2) into (N, 3).

        Only points that are finite in *both* cameras are triangulated; the
        rest are returned as NaN. This avoids paying for the per-point loop
        and undistortPoints work on landmarks that can't be triangulated anyway.
        """
        n = pts0.shape[0]
        result = np.full((n, 3), np.nan, dtype=np.float32)

        both_valid = (
            np.isfinite(pts0).all(axis=1) & np.isfinite(pts1).all(axis=1)
        )
        if not both_valid.any():
            return result

        # Undistort only the valid subset
        v0 = self.undistort(pts0[both_valid], self.k0, self.dist0)
        v1 = self.undistort(pts1[both_valid], self.k1, self.dist1)

        # Batch triangulation: cv.triangulatePoints accepts (2, N) inputs
        # and returns (4, N) homogeneous coords. Much faster than per-point.
        X4 = cv.triangulatePoints(self.p0, self.p1, v0.T, v1.T)  # (4, N_valid)
        w = X4[3]
        # Avoid division by zero
        safe = np.where(np.abs(w) > 1e-12, w, 1.0)
        X3 = (X4[:3] / safe).T  # (N_valid, 3)
        # Any point that had near-zero w becomes NaN
        X3[np.abs(w) <= 1e-12] = np.nan

        result[both_valid] = X3.astype(np.float32)

        if log.isEnabledFor(logging.DEBUG):
            finite = np.isfinite(result).all(axis=1)
            if finite.any():
                log.debug(f"3D range: {result[finite].min():.1f} to {result[finite].max():.1f}")

        return result


def process(pose_json: Path, out_json: Path, tri: Triangulator) -> None:
    """Main driver: read 2-D JSON, load camera params, save 3-D JSON."""
    data = load_json(pose_json)
    pts3d_all = []
    for fr0, fr1 in zip(data["video_0"], data["video_1"]):
        pts0 = np.array([[lm["x"], lm["y"]] for lm in fr0["landmarks"]], dtype=np.float32)
        pts1 = np.array([[lm["x"], lm["y"]] for lm in fr1["landmarks"]], dtype=np.float32)
        pts3d = tri.triangulate(pts0, pts1)
        pts3d_all.extend(pts3d.tolist())

    save_json({"points_3d": pts3d_all}, out_json)
    log.info(f"3D points saved -> {out_json}")