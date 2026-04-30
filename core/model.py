from __future__ import annotations

import logging
import time
import numpy as np
import cv2 as cv

# from pose_estimation.pose2d_extractor import Pose2DEstimator
from pose_estimation.movenet_estimator import MoveNetEstimator
from pose_estimation.triangulate3d import Triangulator
from core.performance_logger import PerformanceLogger
from utils.utils_io import load_extrinsics, load_intrinsics
from collections import deque

log = logging.getLogger(__name__)

SMOOTH_WINDOW=5 # Amount of frames to use for mean value smoothening

class Model:
  def __init__(self, log_performance: bool = False):
    
    # #2D pose estimators
    # self.estimator0 = Pose2DEstimator(model_complexity=0)
    # self.estimator1 = Pose2DEstimator(model_complexity=0)

    self.estimator0 = MoveNetEstimator("lightning")
    self.estimator1 = MoveNetEstimator("lightning")
    
    # Triangulator
    self.triangulator = None
    try:
        k0, _ = load_intrinsics("camera/camera_parameters/camera0_intrinsics.dat")
        k1, _ = load_intrinsics("camera/camera_parameters/camera1_intrinsics.dat")
        r1, t1 = load_extrinsics("camera/camera_parameters/camera1_rot_trans.dat")
        self.triangulator = Triangulator(k0, k1, r1, t1)
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
        
    self._prev_frame0 = None
    self._prev_frame1 = None
        
    # Last keypoints
    self._prev_kp0 = None
    self._prev_kp1 = None

    self._last_result = {
    "frame_left": None,
    "frame_right": None,
    "landmarks_3d": np.full((33, 3), np.nan),
    "landmarks_2d_left": np.full((33, 2), np.nan),
    "landmarks_2d_right": np.full((33, 2), np.nan),
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
  # def process(self, frame0: np.ndarray, frame1: np.ndarray) -> dict:
  #   """
  #   Run a full 3D skeleton generation pipeline from a pair of frames.
    
  #   Args: 
  #     frame0: BGR frame from camera0.
  #     frame1: BGR frame from camera1.
      
  #   Returns dict with:
  #     frame_left:         (H, W, 3) BGR frame.
  #     frame_right:        (H, W, 3) BGR frame.
  #     landmarks_3d:       (33, 3) float32 XYZ in cm, NaN if unavailable.
  #     landmarks_2d_left:  (33, 2) float32 pixel coords cam0.
  #     landmarks_2d_right: (33, 2) float32 pixel coords cam1.
  #   """
  #   # 2D pose estimation
  #   t0 = time.perf_counter()
  #   landmarks_2d_left  = self.estimator0.detect(frame0) 
  #   t1 = time.perf_counter()
  #   landmarks_2d_right = self.estimator1.detect(frame1)
    
  #   # Triangulate to 3D
  #   t2 = time.perf_counter()
  #   landmarks_3d = self.triangulate(landmarks_2d_left, landmarks_2d_right)
  #   t3 = time.perf_counter()
  #   landmarks_3d = self._smooth(landmarks_3d)
  #   t4 = time.perf_counter()
    
  #   self.perf.record(
  #     estimator0_ms  = (t1 - t0) * 1000,
  #     estimator1_ms  = (t2 - t1) * 1000,
  #     triangulate_ms = (t3 - t2) * 1000,
  #     smooth_ms      = (t4 - t3) * 1000,
  #     total_ms       = (t4 - t0) * 1000,
  #     )
    
  #   return {
  #     "frame_left": frame0,
  #     "frame_right": frame1,
  #     "landmarks_3d": landmarks_3d,
  #     "landmarks_2d_left": landmarks_2d_left,
  #     "landmarks_2d_right": landmarks_2d_right,
  #   }

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

        # save for tracking
        self._prev_kp0 = landmarks_2d_left
        self._prev_kp1 = landmarks_2d_right
        self._prev_frame0 = frame0
        self._prev_frame1 = frame1

    # TRACKING
    else:
        if (
            self._prev_kp0 is None
            or self._prev_kp1 is None
            or self._prev_frame0 is None
        ):
            return self._last_result

        landmarks_2d_left  = self._track(self._prev_frame0, frame0, self._prev_kp0)
        t1 = time.perf_counter()
        landmarks_2d_right = self._track(self._prev_frame1, frame1, self._prev_kp1)
        t2 = time.perf_counter()

        # uppdate state
        self._prev_kp0 = landmarks_2d_left
        self._prev_kp1 = landmarks_2d_right
        self._prev_frame0 = frame0
        self._prev_frame1 = frame1

    # Triangulation
    t3 = time.perf_counter()
    landmarks_3d = self.triangulate(landmarks_2d_left, landmarks_2d_right)
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
      return np.full((33, 3), np.nan, dtype=np.float32)
    return self.triangulator.triangulate(pts0, pts1)
  
  # Internal functions
  def _smooth(self, landmarks_3d: np.array) -> np.array:
    """Apply temporal smoothing over the last N frames."""
    self._landmark_buffer.append(landmarks_3d)
    return np.nanmean(self._landmark_buffer, axis=0)
    
