#!/usr/bin/env python3
"""
Stereo-Camera Calibration Utility
=================================

Captures calibration images, computes intrinsic and extrinsic parameters
for two Raspberry Pi CSI cameras and verifies the result visually.

Usage
-----
$ python calibrate.py calibration_settings.yaml
"""

from __future__ import annotations

import os
import sys
import glob
import yaml
import cv2 as cv
import numpy as np
from typing import Dict, Tuple
from scipy import linalg

try:
    from picamera2 import Picamera2
    HAS_PICAMERA = True
except ImportError:
    HAS_PICAMERA = False

# -----------------------------------------------------------------------------#
# Global configuration (loaded from YAML)
# -----------------------------------------------------------------------------#
calibration_settings: Dict[str, any] = {}

# -----------------------------------------------------------------------------#
# I/O helpers
# -----------------------------------------------------------------------------#
def _ensure_dir(path: str) -> None:
    """Create *path* directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def _write_matrix(f, label: str, mat: np.ndarray) -> None:
    """Write a matrix to an open file, one row per line."""
    f.write(f"{label}:\n")
    for row in mat:
        f.write(" ".join(map(str, row)) + "\n")


def _resize_preview(frame: np.ndarray, scale: float) -> np.ndarray:
    """Return a resized copy of *frame* for fast live preview."""
    return cv.resize(frame, None, fx=1 / scale, fy=1 / scale)


# -----------------------------------------------------------------------------#
# Configuration
# -----------------------------------------------------------------------------#
def load_settings(file_name: str) -> None:
    """
    Load calibration parameters from a YAML file into *calibration_settings*.
    Exits with an error message if mandatory keys are missing.
    """
    global calibration_settings

    if not os.path.isfile(file_name):
        sys.exit(f"[ERROR] Settings file not found: {file_name}")

    with open(file_name, "r", encoding="utf-8") as fh:
        calibration_settings = yaml.safe_load(fh)

    mandatory_keys = {"camera0", "camera1", "frame_width", "frame_height"}
    if not mandatory_keys.issubset(calibration_settings):
        sys.exit("[ERROR] Missing keys in YAML. Required: " + ", ".join(mandatory_keys))

    print(f"[INFO] Loaded settings from '{file_name}'")


# -----------------------------------------------------------------------------#
# Geometry utilities
# -----------------------------------------------------------------------------#
def dlt_triangulate(
    P1: np.ndarray, P2: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> np.ndarray:
    A = np.array(
        [
            p1[1] * P1[2, :] - P1[1, :],
            P1[0, :] - p1[0] * P1[2, :],
            p2[1] * P2[2, :] - P2[1, :],
            P2[0, :] - p2[0] * P2[2, :],
        ]
    )
    _, _, Vh = linalg.svd(A.T @ A, full_matrices=False)
    return (Vh[3, 0:3] / Vh[3, 3]).astype(np.float32)


def make_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    H = np.eye(4, dtype=R.dtype)
    H[:3, :3] = R
    H[:3, 3] = t.ravel()
    return H


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return K @ make_homogeneous(R, t)[:3, :]




# -----------------------------------------------------------------------------#
# PiCamera2 wrapper
# -----------------------------------------------------------------------------#
class PiCameraCapture:
    """Wrapper around Picamera2 to mimic cv.VideoCapture interface."""
    def __init__(self, camera_index: int, width: int = 640, height: int = 480):
        self.cam = Picamera2(camera_index)
        config = self.cam.create_preview_configuration(
            main={"format": "RGB888", "size": (width, height)}
        )
        self.cam.configure(config)
        self.cam.start()

    def read(self):
        frame = self.cam.capture_array()
        bgr = cv.cvtColor(frame, cv.COLOR_RGB2XYZ)
        return True, bgr

    def release(self):
        self.cam.stop()
        self.cam.close()

    def isOpened(self):
        return True

# -----------------------------------------------------------------------------#
# Dynamic camera selection
# -----------------------------------------------------------------------------#
def open_camera(camera_id, width=1920, height=1080):
  """
  Try cv.VideoCapture first (USB) with PiCameraCapture as fallback(CSI).
  Returns a camera object with read() and release() interface.
  """
  # Check if USB camera
  if isinstance(camera_id, str) and (camera_id.startswith("/dev/video")):
    cap = cv.VideoCapture(camera_id)
    if cap.isOpened():
      print(f"[CAM] Opened USB Camera {camera_id}")
      return cap
    
  # Fallback to PiCam(CSI)
  if isinstance(camera_id, int):
    if not HAS_PICAMERA:
        print("[WARN] picamera2 not available, cannot open CSI camera")
        return None
    try:
        cap = PiCameraCapture(camera_id)
        return cap
    except Exception as e:
        print(f"[CAM] PiCamera failed: {e}")

    sys.exit(f"[ERROR] could not open camera: {camera_id}")

# -----------------------------------------------------------------------------#
# Image capture — uses PiCameraCapture instead of cv.VideoCapture
# -----------------------------------------------------------------------------#
def capture_single_camera(camera_key: str) -> None:
    """
    Grab checkerboard frames from one CSI camera.
    Saves PNGs into ./frames/.
    """
    _ensure_dir("frames")

    cam_id = calibration_settings[camera_key]
    w = calibration_settings["frame_width"]
    h = calibration_settings["frame_height"]
    n_frames = calibration_settings["mono_calibration_frames"]
    scale = calibration_settings["view_resize"]
    cooldown_default = calibration_settings["cooldown"]

    cap=open_camera(cam_id, width=w, height=h)

    saved, cooldown, recording = 0, cooldown_default, False

    while saved < n_frames:
        ok, frame = cap.read()
        if not ok:
            sys.exit("[ERROR] No data from camera")

        preview = _resize_preview(frame, scale)

        msg = (
            "Press SPACE to start" if not recording else
            f"Cooldown: {cooldown:2d}  |  Saved: {saved}/{n_frames}"
        )
        cv.putText(
            preview, msg, (40, 40), cv.FONT_HERSHEY_COMPLEX, 1,
            (0, 255, 0) if recording else (0, 0, 255), 2
        )

        cv.imshow(f"Preview - {camera_key}", preview)
        key = cv.waitKey(1) & 0xFF

        if key == 27:
            sys.exit("[ABORT] User exit")
        if key == 32:
            recording = True

        if recording:
            cooldown -= 1
            if cooldown <= 0:
                filename = os.path.join("frames", f"{camera_key}_{saved:02d}.png")
                cv.imwrite(filename, frame)
                print(f"[IMG] {filename}")
                saved += 1
                cooldown = cooldown_default

    cap.release()
    cv.destroyAllWindows()


def capture_stereo_pair(cam0: str, cam1: str) -> None:
    """
    Capture synchronized checkerboard frames from cam0 and cam1.
    Saves images to ./frames_pair/.
    """
    _ensure_dir("frames_pair")

    w = calibration_settings["frame_width"]
    h = calibration_settings["frame_height"]
    n_frames = calibration_settings["stereo_calibration_frames"]
    scale = calibration_settings["view_resize"]
    cooldown_default = calibration_settings["cooldown"]

    cap0 = open_camera(calibration_settings[cam0], width=w, height=h)
    cap1 = open_camera(calibration_settings[cam1], width=w, height=h)

    saved, cooldown, recording = 0, cooldown_default, False

    while saved < n_frames:
        ok0, f0 = cap0.read()
        ok1, f1 = cap1.read()
        if not (ok0 and ok1):
            sys.exit("[ERROR] Cameras disconnected")

        p0 = _resize_preview(f0, scale)
        p1 = _resize_preview(f1, scale)

        msg = (
            "SPACE to start" if not recording else
            f"Cooldown: {cooldown:2d}  |  Saved: {saved}/{n_frames}"
        )
        for canvas in (p0, p1):
            cv.putText(canvas, msg, (30, 30), cv.FONT_HERSHEY_COMPLEX, 1,
                       (0, 255, 0) if recording else (0, 0, 255), 2)

        cv.imshow("Cam0", p0)
        cv.imshow("Cam1", p1)
        key = cv.waitKey(1) & 0xFF

        if key == 27:
            sys.exit("[ABORT] User exit")
        if key == 32:
            recording = True

        if recording:
            cooldown -= 1
            if cooldown <= 0:
                fn0 = os.path.join("frames_pair", f"{cam0}_{saved:02d}.png")
                fn1 = os.path.join("frames_pair", f"{cam1}_{saved:02d}.png")
                cv.imwrite(fn0, f0)
                cv.imwrite(fn1, f1)
                print(f"[IMG] {fn0}  |  {fn1}")
                saved += 1
                cooldown = cooldown_default

    cap0.release()
    cap1.release()
    cv.destroyAllWindows()


# -----------------------------------------------------------------------------#
# Calibration helpers
# -----------------------------------------------------------------------------#
def _generate_object_points() -> np.ndarray:
    rows = calibration_settings["checkerboard_rows"]
    cols = calibration_settings["checkerboard_columns"]
    scale = calibration_settings["checkerboard_box_size_scale"]

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    return objp * scale


def calibrate_intrinsics(img_pattern: str) -> Tuple[np.ndarray, np.ndarray]:
    images = sorted(glob.glob(img_pattern))
    if not images:
        sys.exit(f"[ERROR] No images found for pattern {img_pattern}")

    objp = _generate_object_points()
    objpoints, imgpoints = [], []
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-3)

    for fname in images:
        frame = cv.imread(fname)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ok, corners = cv.findChessboardCorners(gray, (calibration_settings["checkerboard_rows"],
                                                       calibration_settings["checkerboard_columns"]), None)

        if ok:
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            objpoints.append(objp)

            preview = frame.copy()
            cv.drawChessboardCorners(preview, (calibration_settings["checkerboard_rows"],
                                               calibration_settings["checkerboard_columns"]), corners, ok)
            cv.putText(preview, "Press 's' to skip this sample",
                       (20, 30), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 1)
            cv.imshow("Check", preview)
            if cv.waitKey(0) & 0xFF == ord("s"):
                objpoints.pop()
                imgpoints.pop()

    cv.destroyAllWindows()

    h, w = cv.imread(images[0]).shape[:2]
    rms, K, dist, *_ = cv.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    print(f"[CALIB] {img_pattern}: RMS = {rms:.4f}")
    print("[CALIB] K =\n", K)
    print("[CALIB] dist =", dist.ravel())
    return K, dist


def save_intrinsics(K: np.ndarray, dist: np.ndarray, cam_key: str) -> None:
    _ensure_dir("camera_parameters")
    with open(f"camera_parameters/{cam_key}_intrinsics.dat", "w") as fh:
        _write_matrix(fh, "intrinsic", K)
        _write_matrix(fh, "distortion", dist)


def stereo_calibrate(
    K0: np.ndarray, d0: np.ndarray,
    K1: np.ndarray, d1: np.ndarray,
    pattern0: str, pattern1: str
) -> Tuple[np.ndarray, np.ndarray]:
    imgs0 = sorted(glob.glob(pattern0))
    imgs1 = sorted(glob.glob(pattern1))
    if not (imgs0 and imgs1 and len(imgs0) == len(imgs1)):
        sys.exit("[ERROR] Stereo frame pairs missing or unsynchronized")

    objp = _generate_object_points()
    objpoints, imgpts_l, imgpts_r = [], [], []
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-3)
    rows = calibration_settings["checkerboard_rows"]
    cols = calibration_settings["checkerboard_columns"]

    for f0, f1 in zip(imgs0, imgs1):
        l_img, r_img = cv.imread(f0), cv.imread(f1)
        g0 = cv.cvtColor(l_img, cv.COLOR_BGR2GRAY)
        g1 = cv.cvtColor(r_img, cv.COLOR_BGR2GRAY)

        ok0, c0 = cv.findChessboardCorners(g0, (rows, cols), None)
        ok1, c1 = cv.findChessboardCorners(g1, (rows, cols), None)
        if not (ok0 and ok1):
            continue

        c0 = cv.cornerSubPix(g0, c0, (11, 11), (-1, -1), criteria)
        c1 = cv.cornerSubPix(g1, c1, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpts_l.append(c0)
        imgpts_r.append(c1)

    h, w = cv.imread(imgs0[0]).shape[:2]
    flags = cv.CALIB_FIX_INTRINSIC
    rms, *_, R, T, _, _ = cv.stereoCalibrate(
        objpoints, imgpts_l, imgpts_r,
        K0, d0, K1, d1, (w, h),
        criteria=criteria, flags=flags
    )

    print(f"[STEREO] RMS = {rms:.4f}")
    print("[STEREO] R =\n", R)
    print("[STEREO] T =\n", T.ravel())
    return R, T


def save_extrinsics(R0: np.ndarray, t0: np.ndarray,
                    R1: np.ndarray, t1: np.ndarray,
                    prefix: str = "") -> None:
    _ensure_dir("camera_parameters")

    with open(f"camera_parameters/{prefix}camera0_rot_trans.dat", "w") as fh:
        _write_matrix(fh, "R", R0)
        _write_matrix(fh, "T", t0)

    with open(f"camera_parameters/{prefix}camera1_rot_trans.dat", "w") as fh:
        _write_matrix(fh, "R", R1)
        _write_matrix(fh, "T", t1)


# -----------------------------------------------------------------------------#
# Calibration sanity-check — uses PiCameraCapture
# -----------------------------------------------------------------------------#
def live_axis_overlay(
    cam0_key: str, cam1_key: str,
    K0: np.ndarray, d0: np.ndarray, R0: np.ndarray, t0: np.ndarray,
    K1: np.ndarray, d1: np.ndarray, R1: np.ndarray, t1: np.ndarray,
    z_shift: float = 50.0,
) -> None:
    """Draw projected XYZ axes on live feeds to visually verify calibration."""
    P0 = projection_matrix(K0, R0, t0)
    P1 = projection_matrix(K1, R1, t1)

    axes = np.array([[0, 0, 0],
                     [1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]], dtype=np.float32)
    axes = 5 * axes + np.array([0, 0, z_shift])

    pix0, pix1 = [], []
    for X in axes:
        Xh = np.append(X, 1)
        pix0.append((P0 @ Xh)[:2] / (P0 @ Xh)[2])
        pix1.append((P1 @ Xh)[:2] / (P1 @ Xh)[2])
    pix0, pix1 = np.int32(pix0), np.int32(pix1)

    w = calibration_settings["frame_width"]
    h = calibration_settings["frame_height"]
    cap0 = open_camera(calibration_settings[cam0_key], width=w, height=h)
    cap1 = open_camera(calibration_settings[cam1_key], width=w, height=h)

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    while True:
        ok0, f0 = cap0.read()
        ok1, f1 = cap1.read()
        if not (ok0 and ok1):
            sys.exit("[ERROR] Video stream interrupted")

        origin = tuple(pix0[0])
        for col, p in zip(colors, pix0[1:]):
            cv.line(f0, origin, tuple(p), col, 2)

        origin = tuple(pix1[0])
        for col, p in zip(colors, pix1[1:]):
            cv.line(f1, origin, tuple(p), col, 2)

        cv.imshow("Camera0", f0)
        cv.imshow("Camera1", f1)
        if cv.waitKey(1) & 0xFF == 27:
            break

    cap0.release()
    cap1.release()
    cv.destroyAllWindows()
    
def run_calibration(settings_path: str = "calibration_settings.yaml") -> bool:
  
    try:
        load_settings(settings_path)
    except SystemExit as e:
        print(f"[ERROR] Could not read settings: {e}")
        return False

    try:
        capture_single_camera("camera0")
        capture_single_camera("camera1")

        K0, d0 = calibrate_intrinsics("frames/camera0*")
        save_intrinsics(K0, d0, "camera0")

        K1, d1 = calibrate_intrinsics("frames/camera1*")
        save_intrinsics(K1, d1, "camera1")

        capture_stereo_pair("camera0", "camera1")

        R01, t01 = stereo_calibrate(
            K0, d0, K1, d1,
            "frames_pair/camera0*", "frames_pair/camera1*"
        )

        R0, t0 = np.eye(3, dtype=np.float32), np.zeros((3, 1), np.float32)
        save_extrinsics(R0, t0, R01, t01)

        print("[INFO] Calibration completed!")
        return True

    except Exception as e:
        print(f"[ERROR] Calibration failed!: {e}")
        return False


# -----------------------------------------------------------------------------#
# Main driver
# -----------------------------------------------------------------------------#
def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python calibrate.py calibration_settings.yaml")

    load_settings(sys.argv[1])

    # Step 1 – capture mono frames
    capture_single_camera("camera0")
    capture_single_camera("camera1")

    # Step 2 – compute intrinsics
    K0, d0 = calibrate_intrinsics("frames/camera0*")
    save_intrinsics(K0, d0, "camera0")

    K1, d1 = calibrate_intrinsics("frames/camera1*")
    save_intrinsics(K1, d1, "camera1")

    # Step 3 – capture stereo pairs
    capture_stereo_pair("camera0", "camera1")

    # Step 4 – stereo calibration
    R01, t01 = stereo_calibrate(
        K0, d0, K1, d1,
        "frames_pair/camera0*", "frames_pair/camera1*"
    )

    # Step 5 – save extrinsics (camera0 is world origin)
    R0, t0 = np.eye(3, dtype=np.float32), np.zeros((3, 1), np.float32)
    save_extrinsics(R0, t0, R01, t01)

    # Optional – live check
    live_axis_overlay(
        "camera0", "camera1",
        K0, d0, R0, t0,
        K1, d1, R01, t01,
        z_shift=60.0
    )


if __name__ == "__main__":
    main()
