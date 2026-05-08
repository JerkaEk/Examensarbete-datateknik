from __future__ import annotations
"""
pose2d_extractor.py
===================

Extract 2-D landmarks from *two* synchronized videos (or dual-lens footage)
and store per-frame landmarks in a single JSON file.

Example
-------
$ python pose2d_extractor.py --v0 cam0.mp4 --v1 cam1.mp4 --out pose2d.json
"""
import argparse
import logging
import os

import cv2 as cv
import mediapipe as mp
import numpy as np

from typing import List, Dict
from pathlib import Path
from utils.utils_io import save_json
from pose_estimation.body_model import NUM_LANDMARKS, mediapipe_to_body

log = logging.getLogger(__name__)
VISIBILITY_THRESHOLD = 0.5

# ──────────────────────────────────────────────────────────────────────────── #
# Mediapipe pose helper
# ──────────────────────────────────────────────────────────────────────────── #
def init_pose() -> mp.solutions.pose.Pose:
    """Return a configured MediaPipe Pose model."""
    return mp.solutions.pose.Pose(model_complexity=2,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

def detect_landmarks(frame, pose) -> List[Dict] | None:
    """Run pose inference on *frame* and return landmark dicts or *None*."""
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    res = pose.process(rgb)
    if not res.pose_landmarks:
        return None
    return [{"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility} for lm in res.pose_landmarks.landmark]

# ──────────────────────────────────────────────────────────────────────────── #
# Main extraction loop
# ──────────────────────────────────────────────────────────────────────────── #
def process(video0: Path, video1: Path, out_json: Path) -> None:
    """Extract landmarks from *video0* and *video1* and dump to *out_json*."""
    cap0, cap1 = (cv.VideoCapture(str(video0)), cv.VideoCapture(str(video1)))
    if not cap0.isOpened() or not cap1.isOpened():
        raise RuntimeError("Video(s) could not be opened")

    pose = init_pose()
    output: Dict[str, List] = {"video_0": [], "video_1": []}
    frame_idx = 0

    while True:
        ok0, f0 = cap0.read()
        ok1, f1 = cap1.read()
        if not (ok0 and ok1):
            break

        # Split stereo frame in half if necessary
        f0_half = f0[:, : f0.shape[1] // 2]
        f1_half = f1[:, f1.shape[1] // 2:]

        for cam_id, frm in enumerate((f0_half, f1_half)):
            lm = detect_landmarks(frm, pose)
            if lm:
                output[f"video_{cam_id}"].append({"frame": frame_idx, "landmarks": lm})

        frame_idx += 1
        if cv.waitKey(1) & 0xFF == 27:   # ESC to abort early
            break

    cap0.release(), cap1.release(), cv.destroyAllWindows()
    save_json(output, out_json)
    log.info(f"Extracted landmarks from {frame_idx} frames -> {out_json}")


# ──────────────────────────────────────────────────────────────────────────── #
# Tasks-API model assets (used when legacy mp.solutions.pose is unavailable)
# ──────────────────────────────────────────────────────────────────────────── #
_MODEL_ASSETS = {
    0: "pose_estimation/pose_landmarker_lite.task",
    1: "pose_estimation/pose_landmarker_full.task",
    2: "pose_estimation/pose_landmarker_heavy.task",
}

def _legacy_pose_available() -> bool:
    try:
        _ = mp.solutions.pose
        return True
    except AttributeError:
        return False


# ──────────────────────────────────────────────────────────────────────────── #
# Live per-frame estimator
# ──────────────────────────────────────────────────────────────────────────── #
class Pose2DEstimator:
    """
    Wraps MediaPipe Pose for live per-frame inference.
    Uses the legacy mp.solutions.pose.Pose API when available (older mediapipe),
    and falls back to the Tasks API PoseLandmarker on newer installs where
    mp.solutions has been removed.

    Input:  BGR or GRAY frame
    Output: (NUM_LANDMARKS, 2) pixel coordinates, NaN if no detection
    """
    def __init__(self, model_complexity: int = 0, **_ignored):
        # **_ignored swallows leftover kwargs (e.g. inference_size, use_roi)
        # from callers that haven't been updated yet.
        self._model_complexity = model_complexity
        self._use_legacy = _legacy_pose_available()

        if self._use_legacy:
            self._pose = mp.solutions.pose.Pose(
                model_complexity=model_complexity,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            log.info("Using legacy MediaPipe Pose API")
        else:
            from mediapipe.tasks import python as _mpt
            from mediapipe.tasks.python import vision as _mpv
            model_path = os.path.abspath(_MODEL_ASSETS.get(model_complexity, _MODEL_ASSETS[0]))
            options = _mpv.PoseLandmarkerOptions(
                base_options=_mpt.BaseOptions(model_asset_path=model_path),
                running_mode=_mpv.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
            )
            self._landmarker = _mpv.PoseLandmarker.create_from_options(options)
            log.info(f"Using Tasks API PoseLandmarker: {model_path}")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if frame.ndim == 2:
            rgb = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
        else:
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        if self._use_legacy:
            res = self._pose.process(rgb)
            if not res.pose_landmarks:
                log.debug("No pose detected in frame")
                return np.full((NUM_LANDMARKS, 2), np.nan, dtype=np.float32)
            raw = np.full((33, 2), np.nan, dtype=np.float32)
            for i, lm in enumerate(res.pose_landmarks.landmark):
                if lm.visibility >= VISIBILITY_THRESHOLD:
                    raw[i] = [lm.x * w, lm.y * h]
            return mediapipe_to_body(raw)
        else:
            data = np.ascontiguousarray(rgb, dtype=np.uint8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=data)
            result = self._landmarker.detect(mp_image)
            if not result or not result.pose_landmarks:
                log.debug("No pose detected in frame")
                return np.full((NUM_LANDMARKS, 2), np.nan, dtype=np.float32)
            raw = np.full((33, 2), np.nan, dtype=np.float32)
            for i, lm in enumerate(result.pose_landmarks[0]):
                if getattr(lm, "visibility", 1.0) >= VISIBILITY_THRESHOLD:
                    raw[i] = [lm.x * w, lm.y * h]
            return mediapipe_to_body(raw)

    def close(self):
        if self._use_legacy:
            self._pose.close()
        else:
            self._landmarker.close()

# ──────────────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0", required=True, help="Left-camera / video-0 file")
    ap.add_argument("--v1", required=True, help="Right-camera / video-1 file")
    ap.add_argument("--out", default="pose2d.json", help="Output JSON file")
    args = ap.parse_args()
    process(Path(args.v0), Path(args.v1), Path(args.out))