from __future__ import annotations
import numpy as np

LANDMARK_NAMES = [
  "left_shoulder", "right_shoulder",
  "left_elbow", "right_elbow",
  "left_wrist", "right_wrist",
  "left_hip", "right_hip",
  "left_knee", "right_knee",
  "left_ankle", "right_ankle",
  "nose",
]

LANDMARK_INDEX = {name: i for i, name in enumerate(LANDMARK_NAMES)}
NUM_LANDMARKS = len(LANDMARK_NAMES)  # 13

CONNECTIONS_NAMED = [
  ("left_shoulder",  "right_shoulder"),
  ("left_shoulder",  "left_elbow"),
  ("left_elbow",     "left_wrist"),
  ("right_shoulder", "right_elbow"),
  ("right_elbow",    "right_wrist"),
  ("left_shoulder",  "left_hip"),
  ("right_shoulder", "right_hip"),
  ("left_hip",       "right_hip"),
  ("left_hip",       "left_knee"),
  ("left_knee",      "left_ankle"),
  ("right_hip",      "right_knee"),
  ("right_knee",     "right_ankle"),
  ("nose",           "left_shoulder"),
  ("nose",           "right_shoulder"),
]

CONNECTIONS = [
  (LANDMARK_INDEX[a], LANDMARK_INDEX[b])
  for a, b in CONNECTIONS_NAMED
]

LANDMARK_GROUPS = {
  "upper_body": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"],
  "lower_body": ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
  "torso":      ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
  "face":       ["nose"],
}

MEDIAPIPE_MAPPING = np.array([11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 0], dtype=int)

def mediapipe_to_body(landmarks_33: np.ndarray) -> np.ndarray:
    return landmarks_33[MEDIAPIPE_MAPPING]

MOVENET_MAPPING = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0], dtype=int)

def movenet_to_body(landmarks_17: np.ndarray) -> np.ndarray:
    return landmarks_17[MOVENET_MAPPING]

def active_indices(groups: list[str]) -> np.ndarray:
    names = set()
    for group in groups:
        names.update(LANDMARK_GROUPS.get(group, []))
    return np.array(sorted(LANDMARK_INDEX[n] for n in names), dtype=int)