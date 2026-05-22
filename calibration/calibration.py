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
import logging
import cv2 as cv
import numpy as np
from typing import Dict, Tuple
from scipy import linalg
from camera.camera_capture import open_camera, PiCameraCapture

log = logging.getLogger(__name__)

# Prevent app from crashing on non-linux machines.
try:
    from picamera2 import Picamera2
    HAS_PICAMERA = True
except ImportError:
    HAS_PICAMERA = False

# Global configuration (loaded from YAML)
calibration_settings: Dict[str, any] = {}

# I/O helpers
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


# Configuration
def load_settings(file_name: str) -> None:
    """
    Load calibration parameters from a YAML file into *calibration_settings*.
    Exits with an error message if mandatory keys are missing.
    """
    # Use a global variable to store calibration settings
    global calibration_settings
    # check if the file exists
    if not os.path.isfile(file_name):
        sys.exit(f"[ERROR] Settings file not found: {file_name}")
    # Load the yaml file and save the settings in calibration_Settings
    with open(file_name, "r", encoding="utf-8") as fh:
        calibration_settings = yaml.safe_load(fh)
    # All mandatory keys (specified in mandatory_keys) must be present in the yaml file
    mandatory_keys = {"camera0", "camera1", "frame_width", "frame_height"}
    if not mandatory_keys.issubset(calibration_settings):
        sys.exit("[ERROR] Missing keys in YAML. Required: " + ", ".join(mandatory_keys))

    log.info(f"Loaded settings from '{file_name}'")


# Geometry utilities
def dlt_triangulate(
    P1: np.ndarray, P2: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> np.ndarray:
    """Triangulate a 3D point from two projection matrices and corresponding 2D points using DLT."""
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
    """Combine rotation matrix R and translation vector t into a 4x4 homogeneous transformation matrix."""
    H = np.eye(4, dtype=R.dtype)
    H[:3, :3] = R
    H[:3, 3] = t.ravel()
    return H


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute the 3x4 projection matrix P = K · [R | t]."""
    return K @ make_homogeneous(R, t)[:3, :]


# Image capture — uses PiCameraCapture instead of cv.VideoCapture
def capture_single_camera(camera_key: str) -> None:
    """
    Grab checkerboard frames from one CSI camera.
    Saves PNGs into ./frames/.
    """
    _ensure_dir("camera/frames")
    # Get the presettings described in the yaml file
    cam_id = calibration_settings[camera_key]
    w = calibration_settings["frame_width"]
    h = calibration_settings["frame_height"]
    n_frames = calibration_settings["mono_calibration_frames"]
    # The scale factor for resizing the preview window. A larger scale value will 
    # result in a smaller preview window, which can help improve performance.
    scale = calibration_settings["view_resize"]
    cooldown_default = calibration_settings["cooldown"]
    # open the camera using the dynamic selection function
    cap=open_camera(cam_id, width=w, height=h)
    saved, cooldown, recording = 0, cooldown_default, False
    # Loop until we have saved the required number of frames
    while saved < n_frames:
        # Read a frame from the camera. If it fails, exit with an error message.
        ok, frame = cap.read()
        if not ok:
            sys.exit("[ERROR] No data from camera")
        # Create a resized preview of the frame for display. This is done to make the display faster.
        preview = _resize_preview(frame, scale)
        # Display instructions or status on the preview image
        msg = (
            "Press SPACE to start" if not recording else
            f"Cooldown: {cooldown:2d}  |  Saved: {saved}/{n_frames}"
        )
        # Write text in the image
        cv.putText(
            preview, msg, (40, 40), cv.FONT_HERSHEY_COMPLEX, 1,
            (0, 255, 0) if recording else (0, 0, 255), 2
        )
        # Show the preview window with the current frame
        cv.imshow(f"Preview - {camera_key}", preview)
        # Wait for a key press for 1 ms
        key = cv.waitKey(1) & 0xFF
        # If the user presses ESC, exit
        if key == 27:
            sys.exit("[ABORT] User exit")
        # If they press SPACE, start recording
        if key == 32:
            recording = True
        
        if recording:
            cooldown -= 1
            # If the cooldown has reached zero, save the current frame
            if cooldown <= 0:
                # Create filename
                filename = os.path.join("camera/frames", f"{camera_key}_{saved:02d}.png")
                # Save the current picture to the specified filename created above
                cv.imwrite(filename, frame)
                log.debug(f"Saved: {filename}")
                # Increment saved with 1
                saved += 1
                cooldown = cooldown_default
    # Stop camera and release resources
    cap.release()
    # Close OpenCV windows
    cv.destroyAllWindows()


def capture_stereo_pair(cam0: str, cam1: str) -> None:
    """
    Capture synchronized checkerboard frames from cam0 and cam1.
    Saves images to ./camera/frames_pair/.
    """
    _ensure_dir("camera/frames_pair")

    # Get the presettings described in the yaml file
    w = calibration_settings["frame_width"]
    h = calibration_settings["frame_height"]
    n_frames = calibration_settings["stereo_calibration_frames"]
    scale = calibration_settings["view_resize"]
    cooldown_default = calibration_settings["cooldown"]
    # open both cameras using open_camera() function
    cap0 = open_camera(calibration_settings[cam0], width=w, height=h)
    cap1 = open_camera(calibration_settings[cam1], width=w, height=h)
    saved, cooldown, recording = 0, cooldown_default, False
    # Loop until we have saved the required number of frames
    while saved < n_frames:
        # Read a frame for each camera
        ok0, f0 = cap0.read()
        ok1, f1 = cap1.read()
        if not (ok0 and ok1):
            sys.exit("[ERROR] Cameras disconnected")
        # Resized preview for display
        p0 = _resize_preview(f0, scale)
        p1 = _resize_preview(f1, scale)
        # Display instructions or status on the preview image
        msg = (
            "SPACE to start" if not recording else
            f"Cooldown: {cooldown:2d}  |  Saved: {saved}/{n_frames}"
        )
        # Write text in the image
        for canvas in (p0, p1):
            cv.putText(canvas, msg, (30, 30), cv.FONT_HERSHEY_COMPLEX, 1,
                       (0, 255, 0) if recording else (0, 0, 255), 2)
        # Show the preview window with the current frame
        cv.imshow("Cam0", p0)
        cv.imshow("Cam1", p1)
        # Wait for a key press for 1 ms
        key = cv.waitKey(1) & 0xFF
        # If the user presses ESC, exit
        if key == 27:
            sys.exit("[ABORT] User exit")
        # If they press SPACE, start recording
        if key == 32:
            recording = True
        if recording:
            cooldown -= 1
            # If the cooldown has reached zero, save the current frame
            if cooldown <= 0:
                # Create filename
                fn0 = os.path.join("camera/frames_pair", f"{cam0}_{saved:02d}.png")
                fn1 = os.path.join("camera/frames_pair", f"{cam1}_{saved:02d}.png")
                # Save the current picture to the specified filename created above
                cv.imwrite(fn0, f0)
                cv.imwrite(fn1, f1)
                log.debug(f"Frame pair: {fn0} - {fn1}")
                # Increment saved with 1
                saved += 1
                cooldown = cooldown_default
    # Stop camera and release resources
    cap0.release()
    cap1.release()
    # Close OpenCV windows
    cv.destroyAllWindows()


def _generate_object_points() -> np.ndarray:
    """Generate 3D checkerboard corner coordinates scaled by box size. Returns (rows*cols, 3) float32 array."""
    rows = calibration_settings["checkerboard_rows"]
    cols = calibration_settings["checkerboard_columns"]
    scale = calibration_settings["checkerboard_box_size_scale"]

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    return objp * scale

def calibrate_intrinsics(img_pattern: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute intrinsic camera parameters from checkerboard images.
    
    Args:
        img_pattern: Global pattern for calibration images e.g. 'camera/frames/camera0*'

    Returns:
        K:    (3, 3) camera matrix
        dist: (N,)  distortion coefficients
    """
    
    # Get all images sorted with matching pattern, for an example camera/frames/camera0*
    images = sorted(glob.glob(img_pattern))
    if not images:
        sys.exit(f"[ERROR] No images found for pattern {img_pattern}")
    # Object points are created based on the information of the checkerboard in the yaml file
    objp = _generate_object_points()
    # objpoints equal to 3D points, imgpoints equal to 2D points in the image
    objpoints, imgpoints = [], []
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-3)
    # Loop through each image
    for fname in images:
        frame = cv.imread(fname)
        # Convert the image to grayscale for better performance in corner detection
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Find the corners of the checkboard pattern. Settings for the pattern are specified in the yaml file
        ok, corners = cv.findChessboardCorners(gray, (calibration_settings["checkerboard_rows"],
                                                       calibration_settings["checkerboard_columns"]), None)
        # if corners are found
        if ok:
            # Refine the corner positions to sub-pixel accuracy for better calibration results
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # Append image points from current image
            imgpoints.append(corners)
            # Append object points, which are the same for all images
            objpoints.append(objp)
            # Make a copy of the frame
            preview = frame.copy()
            # Draw the detected corners on the copy for visualization
            cv.drawChessboardCorners(preview, (calibration_settings["checkerboard_rows"],
                                               calibration_settings["checkerboard_columns"]), corners, ok)
            cv.putText(preview, "Press 's' to skip this sample",
                       (20, 30), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 1)
            # Show the preview and wait for users command
            cv.imshow("Check", preview)
            if cv.waitKey(0) & 0xFF == ord("s"):
                objpoints.pop()
                imgpoints.pop()
                log.debug("Sample skipped by user")

    cv.destroyAllWindows()
    # Get the height and width of images
    h, w = cv.imread(images[0]).shape[:2]
    # Calibrate the camera with object- and image points. Also use image size. 
    # The function returns the re-projection error (rms) and the camera matrix (K) 
    # and distortion of the lens.
    rms, K, dist, *_ = cv.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    log.info(f"Intrinsic calibration complete - RMS: {rms:.4f}")
    log.debug(f"K =\n{K}")
    log.debug(f"dist = {dist.ravel()}")
    # Return the camera matrix and distortion coefficients
    return K, dist


def save_intrinsics(K: np.ndarray, dist: np.ndarray, cam_key: str) -> None:
    """Save camera matrix and distortion coefficients to camera_parameters/{cam_key}_intrinsics.dat."""
    _ensure_dir("camera/camera_parameters")
    with open(f"camera/camera_parameters/{cam_key}_intrinsics.dat", "w") as fh:
        _write_matrix(fh, "intrinsic", K)
        _write_matrix(fh, "distortion", dist)


def stereo_calibrate(
    K0: np.ndarray, d0: np.ndarray,
    K1: np.ndarray, d1: np.ndarray,
    pattern0: str, pattern1: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute extrinsic parameters between two cameras using stereo calibration.

    Returns:
        R: (3, 3) rotation matrix from camera0 to camera1
        T: (3, 1) translation vector from camera0 to camera1
    """
    # Get all sorted image pares 
    imgs0 = sorted(glob.glob(pattern0))
    imgs1 = sorted(glob.glob(pattern1))
    if not (imgs0 and imgs1 and len(imgs0) == len(imgs1)):
        sys.exit("[ERROR] Stereo frame pairs missing or unsynchronized")
    # Object points are created based on the information of the checkerboard in the yaml file
    objp = _generate_object_points()
    # objpoints equal to 3D points, imgpts_l and imgpts_r are 
    # equal to 2D points in the images from left and right camera
    objpoints, imgpts_l, imgpts_r = [], [], []
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-3)
    rows = calibration_settings["checkerboard_rows"]
    cols = calibration_settings["checkerboard_columns"]
    # Loop through each pair
    for f0, f1 in zip(imgs0, imgs1):
        l_img, r_img = cv.imread(f0), cv.imread(f1)
        g0 = cv.cvtColor(l_img, cv.COLOR_BGR2GRAY)
        g1 = cv.cvtColor(r_img, cv.COLOR_BGR2GRAY)
        # Find the corners of the checkboard pattern in each image
        ok0, c0 = cv.findChessboardCorners(g0, (rows, cols), None)
        ok1, c1 = cv.findChessboardCorners(g1, (rows, cols), None)
        if not (ok0 and ok1):
            continue
        # Refine the corner positions to sub-pixel accuracy for better calibration results
        c0 = cv.cornerSubPix(g0, c0, (11, 11), (-1, -1), criteria)
        c1 = cv.cornerSubPix(g1, c1, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpts_l.append(c0)
        imgpts_r.append(c1)
    h, w = cv.imread(imgs0[0]).shape[:2]
    flags = cv.CALIB_FIX_INTRINSIC
    # Calibrate the stereo camera with object- and image points. Also use image size. 
    # The function returns the Rotation (R) and Translation (T) and re-projection error (rms) 
    # and distortion of the lens.
    rms, *_, R, T, _, _ = cv.stereoCalibrate(
        objpoints, imgpts_l, imgpts_r,
        K0, d0, K1, d1, (w, h),
        criteria=criteria, flags=flags
    )
    log.info(f"Extrinsic calibration complete - RMS: {rms:.4f}")
    log.debug(f"R =\n{R}")
    log.debug(f"dist = {T.ravel()}")
    
    
    # Return the roation and translation between the two cameras
    return R, T


def save_extrinsics(R0: np.ndarray, t0: np.ndarray,
                    R1: np.ndarray, t1: np.ndarray,
                    prefix: str = "") -> None:
    """Save rotation and translation matrices for both cameras to camera_parameters/."""
    _ensure_dir("camera/camera_parameters")
    with open(f"camera/camera_parameters/{prefix}camera0_rot_trans.dat", "w") as fh:
        _write_matrix(fh, "R", R0)
        _write_matrix(fh, "T", t0)
    with open(f"camera/camera_parameters/{prefix}camera1_rot_trans.dat", "w") as fh:
        _write_matrix(fh, "R", R1)
        _write_matrix(fh, "T", t1)

    
def run_calibration(settings_path: str = "calibration_settings.yaml") -> bool:
    """
    Run the full calibration pipeline: capture frames, compute intrinsics and extrinsics.

    Returns:
        True if calibration completed successfully, False otherwise.
    """
    try:
        load_settings(settings_path)
    except SystemExit as e:
        log.error(f"Could not read settings: {e}")
        return False
    try:
        # capture frames on each camera
        capture_single_camera("camera0")
        capture_single_camera("camera1")
        # compute intrinsics for camera0 and save
        K0, d0 = calibrate_intrinsics("camera/frames/camera0*")
        save_intrinsics(K0, d0, "camera0")
        # compute intrinsics for camera1 and save
        K1, d1 = calibrate_intrinsics("camera/frames/camera1*")
        save_intrinsics(K1, d1, "camera1")
        # capture stereo pairs for extrinsic calibration
        capture_stereo_pair("camera0", "camera1")
        # compute extrinsics and save
        R01, t01 = stereo_calibrate(
            K0, d0, K1, d1,
            "camera/frames_pair/camera0*", "camera/frames_pair/camera1*"
        )
        R0, t0 = np.eye(3, dtype=np.float32), np.zeros((3, 1), np.float32)
        save_extrinsics(R0, t0, R01, t01)
        log.info("Calibration Completed!")
        return True
    except Exception as e:
        log.error(f"Calibration failed!: {e}")
        return False

def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python calibrate.py calibration_settings.yaml")
    load_settings(sys.argv[1])
    # Step 1 – capture mono frames
    capture_single_camera("camera0")
    capture_single_camera("camera1")
    # Step 2 – compute intrinsics
    K0, d0 = calibrate_intrinsics("camera/frames/camera0*")
    save_intrinsics(K0, d0, "camera0")
    K1, d1 = calibrate_intrinsics("camera/frames/camera1*")
    save_intrinsics(K1, d1, "camera1")
    # Step 3 – capture stereo pairs
    capture_stereo_pair("camera0", "camera1")
    # Step 4 – stereo calibration
    R01, t01 = stereo_calibrate(
        K0, d0, K1, d1,
        "camera/frames_pair/camera0*", "camera/frames_pair/camera1*"
    )
    # Step 5 – save extrinsics (camera0 is world origin)
    R0, t0 = np.eye(3, dtype=np.float32), np.zeros((3, 1), np.float32)
    save_extrinsics(R0, t0, R01, t01)

if __name__ == "__main__":
    main()
