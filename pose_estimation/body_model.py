from __future__ import annotations
import numpy as np

MEDIAPIPE_NUM_LANDMARKS = 33
MOVENET_NUM_LANDMARKS   = 17

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

# (name, index_A, index_B, index_C) — angle measured at B
JOINT_ANGLES = [
  ("Left elbow",      0,  2,  4),   # left_shoulder  → left_elbow    → left_wrist
  ("Right elbow",     1,  3,  5),   # right_shoulder → right_elbow   → right_wrist
  ("Left knee",       6,  8, 10),   # left_hip       → left_knee     → left_ankle
  ("Right knee",      7,  9, 11),   # right_hip      → right_knee    → right_ankle
  ("Left shoulder",   6,  0,  2),   # left_hip       → left_shoulder → left_elbow
  ("Right shoulder",  7,  1,  3),   # right_hip      → right_shoulder → right_elbow
  ("Left hip",        0,  6,  8),   # left_shoulder  → left_hip      → left_knee
  ("Right hip",       1,  7,  9),   # right_shoulder → right_hip     → right_knee
]

def active_indices(groups: list[str]) -> np.ndarray:
    names = set()
    for group in groups:
        names.update(LANDMARK_GROUPS.get(group, []))
    return np.array(sorted(LANDMARK_INDEX[n] for n in names), dtype=int)