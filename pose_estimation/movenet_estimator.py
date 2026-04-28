from __future__ import annotations
"""
movenet_estimator.py
====================
MoveNet-based 2D pose estimator.
Supports two model variants:
  - lightning: faster, less accurate (~30+ FPS)
  - thunder:   slower, more accurate (~15 FPS)
"""
import logging
import numpy as np
import cv2 as cv

from pose_estimation.body_model import (
    NUM_LANDMARKS,
    MOVENET_NUM_LANDMARKS,
    movenet_to_body,
)

log = logging.getLogger(__name__)

MOVENET_NUM_LANDMARKS = 17
CONFIDENCE_THRESHOLD = 0.3  # MoveNet använder score istället för visibility

MODEL_URLS = {
    "lightning": "https://tfhub.dev/google/movenet/singlepose/lightning/4",
    "thunder":   "https://tfhub.dev/google/movenet/singlepose/thunder/4",
}

class MoveNetEstimator:
    """
    Wraps MoveNet for live per-frame inference.
    Input:  BGR frame (H, W, 3)
    Output: (NUM_LANDMARKS, 2) pixel coordinates in body model order, NaN if no detection
    """
    def __init__(self, variant: str = "lightning"):
        """
        Args:
            variant: "lightning" for speed, "thunder" for accuracy
        """
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except ImportError:
            raise ImportError("tensorflow and tensorflow_hub are required for MoveNetEstimator")
        
        if variant not in MODEL_URLS:
            raise ValueError(f"Unknown variant '{variant}', choose 'lightning' or 'thunder'")
        
        self.variant = variant
        log.info(f"Loading MoveNet {variant}...")
        self._model = hub.load(MODEL_URLS[variant])
        self._infer = self._model.signatures["serving_default"]
        log.info(f"MoveNet {variant} loaded")
        
        # Input size per variant
        self._input_size = 192 if variant == "lightning" else 256
        
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run MoveNet inference on a single BGR frame.
        
        Returns:
            (NUM_LANDMARKS, 2) float32 pixel coordinates in body model order.
            NaN for landmarks below confidence threshold.
        """
        try:
            import tensorflow as tf
        except ImportError:
            return np.full((NUM_LANDMARKS, 2), np.nan, dtype=np.float32)
        
        h, w = frame.shape[:2]
        
        # Preprocess — MoveNet expects RGB, square, int32
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        resized = cv.resize(rgb, (self._input_size, self._input_size))
        input_tensor = tf.cast(resized, dtype=tf.int32)
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        
        # Run inference
        outputs = self._infer(input_tensor)
        # Shape: (1, 1, 17, 3) — [y, x, score] normalized 0-1
        keypoints = outputs["output_0"].numpy()[0, 0]  # (17, 3)
        
        # Build full (17, 2) array in pixel coords
        result = np.full((MOVENET_NUM_LANDMARKS, 2), np.nan, dtype=np.float32)
        for i, (y, x, score) in enumerate(keypoints):
            if score >= CONFIDENCE_THRESHOLD:
                result[i] = [x * w, y * h]  # MoveNet returnerar y,x — byt till x,y
        
        return movenet_to_body(result)
    
    def close(self):
        """Release model resources."""
        self._model = None
        self._infer = None
        log.info(f"MoveNet {self.variant} released")