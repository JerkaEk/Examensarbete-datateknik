from __future__ import annotations

import logging
import time
import numpy as np

from pose_estimation.pose2d_extractor import Pose2DEstimator
from pose_estimation.triangulate3d import Triangulator
from core.performance_logger import PerformanceLogger
from pose_estimation.body_model import NUM_LANDMARKS
from utils.utils_io import load_extrinsics, load_intrinsics
from collections import deque

log = logging.getLogger(__name__)

SMOOTH_WINDOW=5 # Amount of frames to use for mean value smoothening

class Model:
  def __init__(self, log_performance: bool = False):
    
    # 2D pose estimators
    self.estimator0 = Pose2DEstimator(model_complexity=0)
    self.estimator1 = Pose2DEstimator(model_complexity=0)
    
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
    
  # Pose Estimation Pipeline
  def process(self, frame0: np.ndarray, frame1: np.ndarray) -> dict:
    """
    Run a full 3D skeleton generation pipeline from a pair of frames.
    
    Args: 
      frame0: BGR frame from camera0.
      frame1: BGR frame from camera1.
      
    Returns dict with:
      frame_left:         (H, W, 3) BGR frame.
      frame_right:        (H, W, 3) BGR frame.
      landmarks_3d:       (13, 3) float32 XYZ in cm, NaN if unavailable.
      landmarks_2d_left:  (13, 2) float32 pixel coords cam0.
      landmarks_2d_right: (13, 2) float32 pixel coords cam1.
    """
    # 2D pose estimation
    t0 = time.perf_counter()
    landmarks_2d_left  = self.estimator0.detect(frame0) 
    t1 = time.perf_counter()
    landmarks_2d_right = self.estimator1.detect(frame1)
    
    # Triangulate to 3D
    t2 = time.perf_counter()
    landmarks_3d = self.triangulate(landmarks_2d_left, landmarks_2d_right)
    t3 = time.perf_counter()
    landmarks_3d = self._smooth(landmarks_3d)
    t4 = time.perf_counter()
    
    self.perf.record(
      estimator0_ms  = (t1 - t0) * 1000,
      estimator1_ms  = (t2 - t1) * 1000,
      triangulate_ms = (t3 - t2) * 1000,
      smooth_ms      = (t4 - t3) * 1000,
      total_ms       = (t4 - t0) * 1000,
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
  def _smooth(self, landmarks_3d: np.array) -> np.array:
    """Apply temporal smoothing over the last N frames."""
    self._landmark_buffer.append(landmarks_3d)
    return np.nanmean(self._landmark_buffer, axis=0)
    
