from __future__ import annotations

import logging
import numpy as np

from pose_estimation.pose2d_extractor import Pose2DEstimator
from pose_estimation.triangulate3d import Triangulator
from utils.utils_io import load_extrinsics, load_intrinsics

log = logging.getLogger(__name__)

class Model:
  def __init__(self):
    
    # 2D pose estimators
    self.estimator0 = Pose2DEstimator(model_complexity=0)
    self.estimator1 = Pose2DEstimator(model_complexity=0)
    
    # Triangulator
    k0, _ = load_intrinsics("camera/camera_parameters/camera0_intrinsics.dat")
    k1, _ = load_intrinsics("camera/camera_parameters/camera1_intrinsics.dat")
    r1, t1 = load_extrinsics("camera/camera_parameters/camera1_rot_trans.dat")
    self.triangulator = Triangulator(k0, k1, r1, t1)
    
  # Cameras
    
  # Calibration
  
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
      landmarks_3d:       (33, 3) float32 XYZ in cm, NaN if unavailable.
      landmarks_2d_left:  (33, 2) float32 pixel coords cam0.
      landmarks_2d_right: (33, 2) float32 pixel coords cam1.
    """
    # 2D pose estimation
    landmarks_2d_left  = self.estimator0.detect(frame0) 
    landmarks_2d_right = self.estimator1.detect(frame1)
    
    # Triangulate to 3D
    landmarks_3d = self.triangulate(landmarks_2d_left, landmarks_2d_right)
    
    return {
      "frame_left": frame0,
      "frame_right": frame1,
      "landmarks_3d": landmarks_3d,
      "landmarks_2d_left": landmarks_2d_left,
      "landmarks_2d_right": landmarks_2d_right,
    }
    

  # Internal functions
  def triangulate(self, pts0: np.ndarray, pts1: np.ndarray) -> np.ndarray:
    """
    Retrieves 3D landmarks from triangulator by sending a pair of 2D landmarks.
    
    """
    if self.triangulator is None:
      return np.full((33, 3), np.nan, dtype=np.float32)
    return self.triangulator.triangulate(pts0, pts1)
    