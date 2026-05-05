from __future__ import annotations
"""
movenet_estimator.py
====================
MoveNet-based 2D pose estimator using TFLite (int8 quantized).
Supports two model variants:
  - lightning: faster, less accurate
  - thunder:   slower, more accurate

On first use, downloads the .tflite model to models/ directory.
Requires tflite_runtime (Pi) or tensorflow (desktop).
"""
import logging
import os
import urllib.request
import numpy as np
import cv2 as cv

from pose_estimation.body_model import (
    NUM_LANDMARKS,
    #MOVENET_NUM_LANDMARKS,
    movenet_to_body,
)

log = logging.getLogger(__name__)

MOVENET_NUM_LANDMARKS = 17
CONFIDENCE_THRESHOLD = 0.3

MODELS_DIR = "models"

MODEL_URLS = {
    "lightning": "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite",
    "thunder":   "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite",
}

MODEL_FILENAMES = {
    "lightning": "movenet_lightning_int8.tflite",
    "thunder":   "movenet_thunder_int8.tflite",
}


def _load_interpreter(model_path: str):
    try:
        import tflite_runtime.interpreter as tflite
        return tflite.Interpreter(model_path=model_path)
    except ImportError:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=model_path)


class MoveNetEstimator:
    """
    Wraps MoveNet TFLite for live per-frame inference.
    Input:  BGR frame (H, W, 3)
    Output: (NUM_LANDMARKS, 2) pixel coordinates in body model order, NaN if no detection
    """
    def __init__(self, variant: str = "lightning"):
        if variant not in MODEL_URLS:
            raise ValueError(f"Unknown variant '{variant}', choose 'lightning' or 'thunder'")

        self.variant = variant
        model_path = self._ensure_model(variant)

        log.info(f"Loading MoveNet {variant} (TFLite)...")
        self._interpreter = _load_interpreter(model_path)
        self._interpreter.allocate_tensors()

        self._input_details  = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self._input_size = self._input_details[0]["shape"][1]  # 192 (lightning) or 256 (thunder)
        log.info(f"MoveNet {variant} loaded (input: {self._input_size}x{self._input_size})")

    def _ensure_model(self, variant: str) -> str:
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = os.path.join(MODELS_DIR, MODEL_FILENAMES[variant])
        if not os.path.exists(path):
            log.info(f"Downloading MoveNet {variant} model to {path}...")
            urllib.request.urlretrieve(MODEL_URLS[variant], path)
            log.info("Download complete")
        return path

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run MoveNet TFLite inference on a single BGR frame.

        Returns:
            (NUM_LANDMARKS, 2) float32 pixel coordinates in body model order.
            NaN for landmarks below confidence threshold.
        """
        h, w = frame.shape[:2]

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        resized = cv.resize(rgb, (self._input_size, self._input_size))
        input_data = np.expand_dims(resized, axis=0).astype(np.uint8)

        self._interpreter.set_tensor(self._input_details[0]["index"], input_data)
        self._interpreter.invoke()

        output = self._interpreter.get_tensor(self._output_details[0]["index"])
        # Dequantize if the model returns int8/uint8
        if self._output_details[0]["dtype"] != np.float32:
            scale, zero_point = self._output_details[0]["quantization"]
            output = (output.astype(np.float32) - zero_point) * scale

        keypoints = output[0, 0]  # (17, 3) — [y, x, score] normalized 0-1

        result = np.full((MOVENET_NUM_LANDMARKS, 2), np.nan, dtype=np.float32)
        for i, (y, x, score) in enumerate(keypoints):
            if score >= CONFIDENCE_THRESHOLD:
                result[i] = [x * w, y * h]

        return movenet_to_body(result)

    def close(self):
        self._interpreter = None
        log.info(f"MoveNet {self.variant} released")
