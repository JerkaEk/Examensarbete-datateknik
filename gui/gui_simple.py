#!/usr/bin/env python3
"""
Real-time dual-camera pose viewer - No GUI framework needed
===========================================================
Uses only OpenCV windows. No PyQt5 or tkinter required.

Dependencies
------------
pip install mediapipe opencv-python numpy

Run
---
$ python3 gui_simple.py --cam0 0 --cam1 1          # live cameras
$ python3 gui_simple.py --cam0 video0.mp4 --cam1 video1.mp4  # video files

Controls
--------
  q / ESC  - quit
  s        - save current frame pair as PNG
"""

from __future__ import annotations
import argparse
import time
import cv2 as cv
import numpy as np
import mediapipe as mp

# ─────────────────────────────────────────────────────────────────── #
# Constants
# ─────────────────────────────────────────────────────────────────── #
DISPLAY_W = 640
DISPLAY_H = 480

CONNECTIONS = [
    (0, 11), (0, 12),   # nose to shoulders
    (11, 13), (13, 15), # left arm
    (12, 14), (14, 16), # right arm
    (11, 23), (12, 24), # shoulders to hips
    (23, 25), (25, 27), # left leg
    (24, 26), (26, 28), # right leg
    (23, 24),           # hips
    (11, 12),           # shoulders
]


# ─────────────────────────────────────────────────────────────────── #
# Helpers
# ─────────────────────────────────────────────────────────────────── #
def open_source(src: str):
    """Open camera index or video file."""
    try:
        return cv.VideoCapture(int(src))
    except ValueError:
        return cv.VideoCapture(src)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv.putText(frame, f"FPS: {fps:.1f}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def draw_skeleton_2d(frame: np.ndarray, landmarks) -> None:
    """Draw 2D skeleton on frame using MediaPipe landmarks."""
    if not landmarks:
        return
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]

    # Draw connections
    for i, j in CONNECTIONS:
        cv.line(frame, pts[i], pts[j], (0, 255, 0), 2)

    # Draw joints
    for pt in pts:
        cv.circle(frame, pt, 4, (0, 0, 255), -1)


def landmarks_to_xyz(result) -> np.ndarray:
    """Convert MediaPipe landmarks to (33, 3) numpy array."""
    if not result.pose_landmarks:
        return np.full((33, 3), np.nan, dtype=np.float32)
    lm = result.pose_landmarks.landmark
    return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)


def make_3d_plot(kps: np.ndarray, size: int = 400) -> np.ndarray:
    """
    Render a simple 3D skeleton as a 2D top-down + side view composite.
    Returns a BGR image of shape (size, size*2, 3).
    """
    canvas = np.zeros((size, size * 2, 3), dtype=np.uint8)

    valid = np.isfinite(kps).all(axis=1)
    if not valid.any():
        cv.putText(canvas, "No pose detected", (10, size // 2),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
        return canvas

    pts = kps[valid]

    # Normalize to [0, 1]
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    rng = np.where((maxs - mins) > 1e-6, maxs - mins, 1.0)
    norm = (kps - mins) / rng  # (33, 3), may contain nan

    margin = 30

    def to_px(u, v, offset_x=0):
        px = int(u * (size - 2 * margin) + margin + offset_x)
        py = int(v * (size - 2 * margin) + margin)
        return px, py

    # Left half: front view (X, Y)
    cv.putText(canvas, "Front (X-Y)", (10, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    for i, j in CONNECTIONS:
        if np.isfinite(norm[i]).all() and np.isfinite(norm[j]).all():
            p1 = to_px(norm[i, 0], norm[i, 1])
            p2 = to_px(norm[j, 0], norm[j, 1])
            cv.line(canvas, p1, p2, (0, 200, 0), 1)
    for i in range(33):
        if np.isfinite(norm[i]).all():
            cv.circle(canvas, to_px(norm[i, 0], norm[i, 1]), 3, (0, 0, 255), -1)

    # Right half: side view (Z, Y)
    cv.putText(canvas, "Side (Z-Y)", (size + 10, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    for i, j in CONNECTIONS:
        if np.isfinite(norm[i]).all() and np.isfinite(norm[j]).all():
            p1 = to_px(norm[i, 2], norm[i, 1], offset_x=size)
            p2 = to_px(norm[j, 2], norm[j, 1], offset_x=size)
            cv.line(canvas, p1, p2, (0, 200, 0), 1)
    for i in range(33):
        if np.isfinite(norm[i]).all():
            cv.circle(canvas, to_px(norm[i, 2], norm[i, 1], offset_x=size),
                      3, (0, 0, 255), -1)

    # Divider line
    cv.line(canvas, (size, 0), (size, size), (80, 80, 80), 1)

    return canvas


# ─────────────────────────────────────────────────────────────────── #
# Main loop
# ─────────────────────────────────────────────────────────────────── #
def run(src0: str, src1: str) -> None:
    cap0 = open_source(src0)
    cap1 = open_source(src1)

    if not cap0.isOpened():
        print(f"[ERR] Could not open source 0: {src0}")
        return
    if not cap1.isOpened():
        print(f"[ERR] Could not open source 1: {src1}")
        return

    print("[INFO] Cameras opened. Press 'q' or ESC to quit, 's' to save frames.")

    # Use model_complexity=0 for best performance on Pi
    pose0 = mp.solutions.pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose1 = mp.solutions.pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    prev_time = time.time()
    frame_count = 0

    while True:
        ok0, f0 = cap0.read()
        ok1, f1 = cap1.read()

        if not (ok0 and ok1):
            print("[INFO] Stream ended or camera disconnected.")
            break

        # Resize for display
        f0 = cv.resize(f0, (DISPLAY_W, DISPLAY_H))
        f1 = cv.resize(f1, (DISPLAY_W, DISPLAY_H))

        # Run MediaPipe
        res0 = pose0.process(cv.cvtColor(f0, cv.COLOR_BGR2RGB))
        res1 = pose1.process(cv.cvtColor(f1, cv.COLOR_BGR2RGB))

        # Draw 2D skeletons
        draw_skeleton_2d(f0, res0.pose_landmarks)
        draw_skeleton_2d(f1, res1.pose_landmarks)

        # FPS
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        draw_fps(f0, fps)

        # Get 3D keypoints
        kps = landmarks_to_xyz(res0)

        # Build display
        cameras = np.hstack([f0, f1])                  # side-by-side cameras
        plot = make_3d_plot(kps, size=DISPLAY_H)        # 3D view

        # Resize plot to match camera height
        plot = cv.resize(plot, (DISPLAY_W, DISPLAY_H))

        # Stack cameras on top, 3D plot below
        top_row = cameras
        bot_row = np.hstack([
            plot,
            np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)  # placeholder
        ])
        display = np.vstack([top_row, bot_row])

        cv.imshow("Pose Viewer  [q=quit  s=save]", display)

        key = cv.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q or ESC
            break
        elif key == ord('s'):
            fname = f"frame_{frame_count:04d}.png"
            cv.imwrite(fname, display)
            print(f"[SAVED] {fname}")

        frame_count += 1

    cap0.release()
    cap1.release()
    cv.destroyAllWindows()
    print("[INFO] Done.")


# ─────────────────────────────────────────────────────────────────── #
# Entry point
# ─────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dual-camera pose viewer")
    ap.add_argument("--cam0", default="0", help="Camera 0 index or video file (default: 0)")
    ap.add_argument("--cam1", default="1", help="Camera 1 index or video file (default: 1)")
    args = ap.parse_args()

    run(args.cam0, args.cam1)
