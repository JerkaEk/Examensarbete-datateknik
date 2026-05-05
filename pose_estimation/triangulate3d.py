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
         # set camera0 in origo, with no rotation or translation
         r0, t0 = np.eye(3), np.zeros((3, 1))
         self.k0 = k0
         self.k1 = k1
         self.dist0 = dist0
         self.dist1 = dist1
         self.p0 = self.build_projection(k0, r0, t0)
         self.p1 = self.build_projection(k1, r1, t1)

    def build_projection(self, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Return 3×4 projection matrix P = K · [R | t]."""
        Rt = np.hstack((R, t))
        return K @ Rt
    
    def undistort(self, pts, K, dist):
        if dist is None:
            return pts
        pts = pts.reshape(-1, 1, 2)
        undist = cv.undistortPoints(pts, K, dist, P=K)
        return undist.reshape(-1, 2)
    
    def triangulate(self, pts0: np.ndarray, pts1: np.ndarray) -> np.ndarray:
        """Triangulate each pair of points (Nx2, Nx2) into (x, y, z)."""
        pts_3d = []
        pts0 = self.undistort(pts0, self.k0, self.dist0)
        pts1 = self.undistort(pts1, self.k1, self.dist1)
        for p0, p1 in zip(pts0, pts1):
            if np.isnan(p0).any() or np.isnan(p1).any():
                pts_3d.append([np.nan, np.nan, np.nan])
                continue
            X4 = cv.triangulatePoints(self.p0, self.p1, p0.reshape(2, 1), p1.reshape(2, 1))
            X3 = (X4[:3] / X4[3]).ravel()
            pts_3d.append(X3)
        result = np.array(pts_3d, dtype=np.float32)
        valid = np.isfinite(result).all(axis=1)
        if valid.any():
            log.debug(f"3D range: {result[valid].min():.1f} to {result[valid].max():.1f}")
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
    print(f"[3-D] saved → {out_json}")
