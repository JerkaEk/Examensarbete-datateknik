from __future__ import annotations

import logging
import time
import numpy as np
import cv2 as cv

from concurrent.futures import ThreadPoolExecutor
from pose_estimation.triangulate3d import Triangulator
from pose_estimation.body_model import NUM_LANDMARKS, JOINT_ANGLES
from core.performance_logger import PerformanceLogger
from utils.utils_io import load_extrinsics, load_intrinsics
from collections import deque

log = logging.getLogger(__name__)

from pose_estimation.mediapipe_estimator import Pose2DEstimator as _Estimator
_ESTIMATOR_KWARGS: dict = {"model_complexity": 0}
# from pose_estimation.movenet_estimator import MoveNetEstimator as _Estimator
# _ESTIMATOR_KWARGS: dict = {"variant": "lightning"}

SMOOTH_WINDOW = 5


def _to_gray(frame: np.ndarray) -> np.ndarray:
    return frame if frame.ndim == 2 else cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


def _timed(fn, *args):
    """Run fn(*args) and return (result, elapsed_ms)."""
    s = time.perf_counter()
    r = fn(*args)
    return r, (time.perf_counter() - s) * 1000.0


class Model:
    def __init__(self, log_performance: bool = False, perf: PerformanceLogger = None):

        # 2D pose estimators (one per camera — MediaPipe Pose is not thread-safe to share)
        self.estimator0 = _Estimator(**_ESTIMATOR_KWARGS)
        self.estimator1 = _Estimator(**_ESTIMATOR_KWARGS)

        # Thread pool for parallel left/right inference + tracking.
        # MediaPipe and OpenCV release the GIL during heavy work, so threads give real parallelism.
        self._pool = ThreadPoolExecutor(max_workers=2)

        # Triangulator
        self.triangulator = None
        try:
            k0, dist0 = load_intrinsics("camera/camera_parameters/camera0_intrinsics.dat")
            k1, dist1 = load_intrinsics("camera/camera_parameters/camera1_intrinsics.dat")
            r1, t1 = load_extrinsics("camera/camera_parameters/camera1_rot_trans.dat")
            self.triangulator = Triangulator(k0, k1, dist0, dist1, r1, t1)
            log.info("Calibration loaded successfully")
        except FileNotFoundError:
            log.warning("Calibration files not found")

        self._landmark_buffer = deque(maxlen=SMOOTH_WINDOW)
        self.perf = perf if perf is not None else PerformanceLogger(log_to_csv=log_performance)

        ## Inference and tracking
        self._frame_idx = 0
        self._inference_interval = 1  # 1 = inference every frame, 2 = every other, ...

        # Cached grayscale for the most recently processed frames.
        # Filled both by inference (so the next tracking call can reuse it)
        # and by tracking (so the call after that doesn't recompute it).
        self._prev_gray0 = None
        self._prev_gray1 = None
        self._prev_kp0 = None
        self._prev_kp1 = None

        self._last_result = {
            "frame_left": None,
            "frame_right": None,
            "landmarks_3d": np.full((NUM_LANDMARKS, 3), np.nan),
            "landmarks_2d_left": np.full((NUM_LANDMARKS, 2), np.nan),
            "landmarks_2d_right": np.full((NUM_LANDMARKS, 2), np.nan),
        }

    def close(self):
        """Shut down worker threads and estimators cleanly."""
        self._pool.shutdown(wait=True)
        try:
            self.estimator0.close()
            self.estimator1.close()
        except AttributeError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ------------------------------------------------------------------ #
    # Tracking
    # ------------------------------------------------------------------ #
    def _track(self, prev_gray, curr_gray, keypoints):
        if keypoints is None:
            return None

        valid = ~np.isnan(keypoints[:, 0])
        pts_prev = keypoints[valid].reshape(-1, 1, 2).astype(np.float32)

        if len(pts_prev) == 0:
            return keypoints

        pts_next, status, _ = cv.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts_prev, None
        )

        new_kp = keypoints.copy()
        idx = 0
        for i, v in enumerate(valid):
            if v:
                if status[idx][0] == 1:
                    new_kp[i] = pts_next[idx][0]
                idx += 1
        return new_kp

    # ------------------------------------------------------------------ #
    # Pose Estimation Pipeline
    # ------------------------------------------------------------------ #
    def process(self, frame0: np.ndarray, frame1: np.ndarray) -> dict:
        """Pipeline with staggered inference + tracking, left/right run in parallel."""

        t_start = time.perf_counter()
        self._frame_idx += 1

        # Compute grayscale at most once per frame, and only when we'll actually
        # need it. Inference branch needs it to prime the next tracking call;
        # tracking branch needs it as the "current" gray.
        gray0 = None
        gray1 = None

        # ---- INFERENCE (parallel) ----
        if self._frame_idx % self._inference_interval == 0:
            fut_left  = self._pool.submit(_timed, self.estimator0.detect, frame0)
            fut_right = self._pool.submit(_timed, self.estimator1.detect, frame1)

            landmarks_2d_left,  est0_ms = fut_left.result()
            landmarks_2d_right, est1_ms = fut_right.result()

            # Prime the gray cache so the next tracking call can use it.
            gray0 = _to_gray(frame0)
            gray1 = _to_gray(frame1)

            self._prev_kp0 = landmarks_2d_left
            self._prev_kp1 = landmarks_2d_right
            self._prev_gray0 = gray0
            self._prev_gray1 = gray1

        # ---- TRACKING (parallel) ----
        else:
            if self._prev_kp0 is None or self._prev_kp1 is None or self._prev_gray0 is None:
                return self._last_result

            gray0 = _to_gray(frame0)
            gray1 = _to_gray(frame1)

            fut_left  = self._pool.submit(_timed, self._track, self._prev_gray0, gray0, self._prev_kp0)
            fut_right = self._pool.submit(_timed, self._track, self._prev_gray1, gray1, self._prev_kp1)

            landmarks_2d_left,  est0_ms = fut_left.result()
            landmarks_2d_right, est1_ms = fut_right.result()

            self._prev_kp0 = landmarks_2d_left
            self._prev_kp1 = landmarks_2d_right
            self._prev_gray0 = gray0
            self._prev_gray1 = gray1

        # ---- Early-out if both 2D results are entirely NaN ----
        left_all_nan  = landmarks_2d_left  is None or np.isnan(landmarks_2d_left).all()
        right_all_nan = landmarks_2d_right is None or np.isnan(landmarks_2d_right).all()
        if left_all_nan and right_all_nan:
            landmarks_3d = np.full((NUM_LANDMARKS, 3), np.nan, dtype=np.float32)
            t3 = t4 = t5 = time.perf_counter()
        else:
            # ---- Triangulation ----
            t3 = time.perf_counter()
            landmarks_3d = self.triangulate(landmarks_2d_left, landmarks_2d_right)
            landmarks_3d = self._filter_outliers(landmarks_3d)
            t4 = time.perf_counter()

            # ---- Smoothing ----
            landmarks_3d = self._smooth(landmarks_3d)
            t5 = time.perf_counter()

        # ---- Performance log ----
        self.perf.record(
            estimator0_ms  = est0_ms,
            estimator1_ms  = est1_ms,
            triangulate_ms = (t4 - t3) * 1000.0,
            smooth_ms      = (t5 - t4) * 1000.0,
            total_ms       = (t5 - t_start) * 1000.0,
        )

        result = {
            "frame_left":         frame0,
            "frame_right":        frame1,
            "landmarks_3d":       landmarks_3d,
            "landmarks_2d_left":  landmarks_2d_left,
            "landmarks_2d_right": landmarks_2d_right,
            "joint_angles":       self._compute_angles(landmarks_3d),
        }
        self._last_result = result
        return result

    # ------------------------------------------------------------------ #
    # Triangulation
    # ------------------------------------------------------------------ #
    def triangulate(self, pts0: np.ndarray, pts1: np.ndarray) -> np.ndarray:
        """Retrieves 3D landmarks from triangulator by sending a pair of 2D landmarks."""
        if self.triangulator is None:
            return np.full((NUM_LANDMARKS, 3), np.nan, dtype=np.float32)
        return self.triangulator.triangulate(pts0, pts1)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _filter_outliers(self, landmarks_3d: np.ndarray, threshold_mm: float = 1000.0) -> np.ndarray:
        """Set landmarks to NaN if they are more than threshold_mm from the median of valid landmarks."""
        valid = np.isfinite(landmarks_3d).all(axis=1)
        if valid.sum() < 3:
            return landmarks_3d
        median = np.median(landmarks_3d[valid], axis=0)
        dists = np.linalg.norm(landmarks_3d - median, axis=1)
        result = landmarks_3d.copy()
        result[valid & (dists > threshold_mm)] = np.nan
        return result

    def _compute_angles(self, landmarks_3d: np.ndarray) -> dict:
        angles = {}
        for name, ai, bi, ci in JOINT_ANGLES:
            a, b, c = landmarks_3d[ai], landmarks_3d[bi], landmarks_3d[ci]
            if np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(c).all():
                ba, bc = a - b, c - b
                n = np.linalg.norm(ba) * np.linalg.norm(bc)
                if n > 0:
                    angles[name] = float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / n, -1.0, 1.0))))
        return angles

    def _smooth(self, landmarks_3d: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing over the last N frames."""
        if self._landmark_buffer and self._landmark_buffer[0].shape != landmarks_3d.shape:
            self._landmark_buffer.clear()
        self._landmark_buffer.append(landmarks_3d)
        return np.nanmean(self._landmark_buffer, axis=0)