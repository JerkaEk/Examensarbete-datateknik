from __future__ import annotations

import logging
import time
import numpy as np
import cv2 as cv

from pose_estimation.triangulate3d import Triangulator
from pose_estimation.body_model import NUM_LANDMARKS
from core.performance_logger import PerformanceLogger
from utils.utils_io import load_extrinsics, load_intrinsics
from collections import deque

log = logging.getLogger(__name__)

from pose_estimation.pose2d_extractor import Pose2DEstimator as _Estimator
_ESTIMATOR_KWARGS: dict = {"model_complexity": 0}
# from pose_estimation.movenet_estimator import MoveNetEstimator as _Estimator
# _ESTIMATOR_KWARGS: dict = {"variant": "lightning"}

SMOOTH_WINDOW = 5

def _to_gray(frame: np.ndarray) -> np.ndarray:
    return frame if frame.ndim == 2 else cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

class Model:
  def __init__(self, log_performance: bool = False):
    
    # 2D pose estimators
    self.estimator0 = _Estimator(**_ESTIMATOR_KWARGS)
    self.estimator1 = _Estimator(**_ESTIMATOR_KWARGS)
    
    # Triangulator
    self.triangulator = None
    try:
        k0, dist0 = load_intrinsics("camera/camera_parameters/camera0_intrinsics.dat")
        k1, dist1 = load_intrinsics("camera/camera_parameters/camera1_intrinsics.dat")
        r1, t1 = load_extrinsics("camera/camera_parameters/camera1_rot_trans.dat")
        self.triangulator = Triangulator(k0, k1, dist0, dist1, r1, t1)
        log.info("Calibration loaded successfully")
    except FileNotFoundError:
        log.warning(f"Calibration files not found")
        
    self._landmark_buffer = deque(maxlen=SMOOTH_WINDOW)
    
    self.perf = PerformanceLogger(log_to_csv=log_performance)

    ## Inference och tracking
    # Frame counter
    self._frame_idx = 0
        
    # Do inference for every N frame
    self._inference_interval = 2  #  50% 
        
    # Latest result for fallback
    self._last_result = None
        
    self._prev_gray0 = None
    self._prev_gray1 = None

    # Last keypoints
    self._prev_kp0 = None
    self._prev_kp1 = None

    self._last_result = {
    "frame_left": None,
    "frame_right": None,
    "landmarks_3d": np.full((NUM_LANDMARKS, 3), np.nan),
    "landmarks_2d_left": np.full((NUM_LANDMARKS, 2), np.nan),
    "landmarks_2d_right": np.full((NUM_LANDMARKS, 2), np.nan),
}
  
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
    
  # Pose Estimation Pipeline
  def process(self, frame0: np.ndarray, frame1: np.ndarray) -> dict:
    """pipeline with staggered inference + tracking
    """

    t0 = time.perf_counter()
    self._frame_idx += 1

    #  INFERENCE
    if self._frame_idx % self._inference_interval == 0:
        landmarks_2d_left  = self.estimator0.detect(frame0)
        t1 = time.perf_counter()
        landmarks_2d_right = self.estimator1.detect(frame1)
        t2 = time.perf_counter()

        self._prev_kp0 = landmarks_2d_left
        self._prev_kp1 = landmarks_2d_right
        self._prev_gray0 = _to_gray(frame0)
        self._prev_gray1 = _to_gray(frame1)

    # TRACKING
    else:
        if self._prev_kp0 is None or self._prev_kp1 is None or self._prev_gray0 is None:
            return self._last_result

        gray0 = _to_gray(frame0)
        gray1 = _to_gray(frame1)
        landmarks_2d_left  = self._track(self._prev_gray0, gray0, self._prev_kp0)
        t1 = time.perf_counter()
        landmarks_2d_right = self._track(self._prev_gray1, gray1, self._prev_kp1)
        t2 = time.perf_counter()

        self._prev_kp0 = landmarks_2d_left
        self._prev_kp1 = landmarks_2d_right
        self._prev_gray0 = gray0
        self._prev_gray1 = gray1
    
    # Triangulation
    t3 = time.perf_counter()
    landmarks_3d = self.triangulate(landmarks_2d_left, landmarks_2d_right)
    landmarks_3d = self._filter_outliers(landmarks_3d)
    t4 = time.perf_counter()

    # Smoothing
    landmarks_3d = self._smooth(landmarks_3d)
    t5 = time.perf_counter()

    #  Performance log 
    self.perf.record(
        estimator0_ms  = (t1 - t0) * 1000,
        estimator1_ms  = (t2 - t1) * 1000,
        triangulate_ms = (t4 - t3) * 1000,
        smooth_ms      = (t5 - t4) * 1000,
        total_ms       = (t5 - t0) * 1000,
    )

    return {
        "frame_left": frame0,
        "frame_right": frame1,
        "landmarks_3d": landmarks_3d,
        "landmarks_2d_left": landmarks_2d_left,
        "landmarks_2d_right": landmarks_2d_right,
    }
    
  def triangulate(self, pts0: np.ndarray, pts1: np.ndarray) -> np.ndarray:
    """
    Retrieves 3D landmarks from triangulator by sending a pair of 2D landmarks.
    
    """
    if self.triangulator is None:
      return np.full((NUM_LANDMARKS, 3), np.nan, dtype=np.float32)
    return self.triangulator.triangulate(pts0, pts1)
  
  # Internal functions
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

  def _smooth(self, landmarks_3d: np.ndarray) -> np.ndarray:
    """Apply temporal smoothing over the last N frames."""
    if self._landmark_buffer and self._landmark_buffer[0].shape != landmarks_3d.shape:
      self._landmark_buffer.clear()
    self._landmark_buffer.append(landmarks_3d)
    return np.nanmean(self._landmark_buffer, axis=0)
    
