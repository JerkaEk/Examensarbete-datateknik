import logging
import time
import queue
import threading

import cv2 as cv

from camera.camera_capture import open_camera
from core.model import Model
from utils.utils_io import load_yaml, save_yaml, create_video_writers
from core.performance_logger import PerformanceLogger
from datetime import datetime

log = logging.getLogger(__name__)

PREVIEW_INTERVAL_MS = 16
PREVIEW_SCALE = 2  # Downsample factor for preview frames (pre-processed in reader thread)
CALIBRATION_SETTINGS_PATH = "calibration/calibration_settings.yaml"


class _CameraReader:
    """Reads frames in a background thread, always exposing the latest frame.

    Maintains two buffers:
      - _frame:   full-resolution BGR for the model
      - _preview: half-resolution RGB numpy for the GUI preview
    Heavy work (resize + cvtColor) happens here, not in the GUI thread.
    """
    def __init__(self, cap, is_video: bool = False):
        self._cap = cap
        self._frame = None
        self._preview = None
        self._ret = False
        self._lock = threading.Lock()
        self._cap_lock = threading.Lock()
        self._stop = threading.Event()
        self._capture_fps = 0.0
        self._capture_time = 0.0
        self._cap_count = 0
        self._cap_t0 = time.time()
        self._frame_version=0
        # Frame pacing is only needed for video files. For live cameras cap.read()
        # already blocks until the hardware delivers the next frame.
        if is_video:
            fps = cap.get(cv.CAP_PROP_FPS)
            self._frame_interval = 1.0 / fps if fps > 0 else 0.0
        else:
            self._frame_interval = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                t = time.time()

                with self._cap_lock:
                    ret, frame = self._cap.read()

                capture_time = time.perf_counter()

                if ret:
                    h, w = frame.shape[:2]
                    small = cv.resize(frame, (w // PREVIEW_SCALE, h // PREVIEW_SCALE))

                    if small.ndim == 2:
                        preview = cv.cvtColor(small, cv.COLOR_GRAY2RGB)
                    else:
                        preview = cv.cvtColor(small, cv.COLOR_BGR2RGB)

                    with self._lock:
                        self._ret = True
                        self._frame = frame
                        self._preview = preview
                        self._capture_time = capture_time
                        self._frame_version += 1

                    # FPS (only valid frames)
                    self._cap_count += 1
                    elapsed_fps = time.time() - self._cap_t0
                    if elapsed_fps >= 2.0:
                        self._capture_fps = self._cap_count / elapsed_fps
                        self._cap_count = 0
                        self._cap_t0 = time.time()

                    # Frame pacing
                    elapsed = time.time() - t
                    wait = self._frame_interval - elapsed
                    if wait > 0:
                        time.sleep(wait)

                else:
                    with self._lock:
                        self._ret = False
                    time.sleep(0.01)  # avoid busy loop

            except Exception:
                log.exception("_CameraReader loop error")

    @property
    def capture_fps(self) -> float:
        return self._capture_fps

    def read(self):
        with self._lock:
            return self._ret, self._frame, self._capture_time, self._frame_version

    def read_preview(self):
        with self._lock:
            return self._preview, self._frame_version

    def get(self, prop):
        with self._cap_lock:
            return self._cap.get(prop)

    def set(self, prop, value):
        with self._cap_lock:
            return self._cap.set(prop, value)

    def release(self):
        self._stop.set()
        with self._cap_lock:
            self._cap.release()

    def isOpened(self):
        with self._cap_lock:
            return self._cap.isOpened()

class Controller:
    """Main controller for the Stereo 3D Skeleton Tracker."""
    def __init__(self, no_gui=False, mock_cameras=False):   
        self.no_gui = no_gui
        self.mock_cameras = mock_cameras

        # State
        self.mode = "camera"    # camera/video
        self.preview_active = False
        self._estimation_active = False
        self._show_preview = True
        self.gui = None
        self.calibration_running = False
        
        # Cameras
        self.cam0_id = None
        self.cam1_id = None
        self.cam0 = None
        self.cam1 = None
        
        # Video
        self.video0_path = None
        self.video1_path = None
        
        self._last_frame_version0 = -1
        self._last_frame_version1 = -1
        self._last_preview_version0 = -1
        self._last_preview_version1 = -1
        
        # Recording
        self._writer0 = None
        self._writer1 = None
        self._recording = False

        # Shared performance logger — model and controller write to the same CSV
        self.perf = PerformanceLogger(log_to_csv=True)
        self.model = Model(log_performance=True, perf=self.perf)
        
        log.debug("Controller initialized")
    
    def run(self):
        """Start the application by launching GUI"""
        log.info("Starting application")
        self._start_gui()
    
    def shutdown(self):
        """Gracefully shut down cameras and threads."""
        log.info("Shutting down")
        if self.preview_active:
            self._stop_preview()
            
        if self._recording:
            self.stop_recording()
            
        self.perf.close()  # model.perf is the same object — close only once
        
        # Close all process threads here.     

    def on_calibrate_clicked(self):
        """Called by GUI when user clicks Calibrate."""
        if self.calibration_running:
            log.warning("Calibration already running")
            return

        if self.preview_active:
            self._stop_preview()

        log.info("Starting calibration")
        self.calibration_running = True

        from calibration.calibration import run_calibration
        success = run_calibration(CALIBRATION_SETTINGS_PATH)

        if success:
            log.info("Calibration complete")
            self.gui.show_calibration_status("Cameras calibrated")
        else:
            log.error("Calibration failed")

        self.calibration_running = False
        if self.cam0_id and self.cam1_id:
            self._start_preview()
        
    def on_calibration_settings_opened(self):
        """Called by GUI when calibration settings panel opens — loads yaml into GUI."""
        log.debug("Calibration settings opened")
        values = load_yaml(CALIBRATION_SETTINGS_PATH)
        self.gui.populate_settings(values)
        
    def on_calibration_settings_saved(self, values: dict):
        """Called by GUI when user saves calibration settings."""
        log.info("Saving calibration settings")
        settings = load_yaml(CALIBRATION_SETTINGS_PATH)
        for key, value in values.items():
            try:
                settings[key] = int(value)
            except (ValueError, TypeError):
                try:
                    settings[key] = float(value)
                except (ValueError, TypeError):
                    settings[key] = value
        
       # settings.update(values)
        save_yaml(settings, CALIBRATION_SETTINGS_PATH)
        
    def on_toggle_preview(self):
        """Toggle live preview on/off."""
        if self.preview_active:
            log.info("Stopping preview")
            self._stop_preview()
        else:
            log.info("Starting preview")
            self._start_preview()
        
    def on_scan_cameras(self):
        """Detect cameras and send result to GUI."""
        from camera.detect_cameras import detect_all_cameras
        cameras = detect_all_cameras()    
        self.gui.update_camera_list(cameras)
        
    def on_video_paths_saved(self, path0: str, path1: str):
        """Called by GUI when user selects video files."""
        self.video0_path = path0
        self.video1_path = path1
        self.mode = "video"
        log.info(f"Video mode: {path0}, {path1}")
        if self.preview_active:
            self._stop_preview()
        self._start_preview()
        
    def on_switch_to_camera_mode(self):
        """Called by GUI when user switches back to camera mode."""
        self.mode = "camera"
        if self.preview_active:
            self._stop_preview()
        if self.cam0_id and self.cam1_id:
            self._start_preview()
            
    def start_recording(self):
        """Start recording frames from both cameras."""
        w0 = int(self.cam0.get(cv.CAP_PROP_FRAME_WIDTH))
        h0 = int(self.cam0.get(cv.CAP_PROP_FRAME_HEIGHT))
        w1 = int(self.cam1.get(cv.CAP_PROP_FRAME_WIDTH))
        h1 = int(self.cam1.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = self._poll_fps

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._writer0, self._writer1 = create_video_writers(timestamp, (w0, h0), (w1, h1), fps=fps)
        self._recording = True
        log.info(f"Recording started: {timestamp} @ {fps:.0f}fps")
        
    def stop_recording(self):
        """Stop recording and release writers."""
        self._recording = False
        if self._writer0:
            self._writer0.release()
            self._writer0 = None
        if self._writer1:
            self._writer1.release()
            self._writer1 = None
        log.info("Recording stopped")
        
    # Send to GUI
    
    # Utilities
    
    def on_camera_settings_saved(self, cam0: dict, cam1: dict):
        """
        Called by GUI when user saves camera selection.
        """
        log.info(f"Selected cameras: cam0={cam0['display']}, cam1={cam1['display']}")
        self.cam0_id = cam0['id']
        self.cam1_id = cam1['id']
        
        settings = load_yaml(CALIBRATION_SETTINGS_PATH)
        settings["camera0"] = cam0["id"]
        settings["camera1"] = cam1["id"]
        settings["camera0_display"] = cam0["display"]
        settings["camera1_display"] = cam1["display"]
        save_yaml(settings, CALIBRATION_SETTINGS_PATH)
        
        self._start_preview()
        
    def _restore_camera_selection(self):
        settings = load_yaml(CALIBRATION_SETTINGS_PATH)
        cam0_id = settings.get("camera0")
        cam1_id = settings.get("camera1")
        cam0_display = settings.get("camera0_display")
        cam1_display = settings.get("camera1_display")

        if not all([cam0_id, cam1_id, cam0_display, cam1_display]):
            log.debug("No saved camera selection found")
            return

        cam0 = {"id": cam0_id, "display": cam0_display}
        cam1 = {"id": cam1_id, "display": cam1_display}
        
        self.cam0_id = cam0_id
        self.cam1_id = cam1_id
        self.gui.preselect_cameras(cam0, cam1)
        self._start_preview()
    
    def _start_preview(self):
        settings = load_yaml(CALIBRATION_SETTINGS_PATH)
        w = settings.get("frame_width", 640)
        h = settings.get("frame_height", 360)
        fps = settings.get("frame_fps", 120)

        is_video = self.mode == "video"
        if is_video:
            cam0_id = self.video0_path
            cam1_id = self.video1_path
        else:
            cam0_id = self.cam0_id
            cam1_id = self.cam1_id

        log.info(f"Opening: {cam0_id}, {cam1_id}")
        cap0 = open_camera(cam0_id, width=w, height=h, fps=fps)
        cap1 = open_camera(cam1_id, width=w, height=h, fps=fps)

        if not cap0 or not cap1:
            log.error("Could not open cameras/videos")
            return

        self.cam0 = _CameraReader(cap0, is_video=is_video)
        self.cam1 = _CameraReader(cap1, is_video=is_video)
        self.preview_active = True
        self._frame_queue = queue.Queue(maxsize=2)
        self._fps_count = 0
        self._fps_t0 = time.time()
        self._last_model_fps = 0

        self._poll_fps = 30.0  # updated once per 2s; used as VideoWriter FPS

        self._model_thread = threading.Thread(target=self._model_loop, daemon=True)
        self._model_thread.start()
        # Give the reader threads time to grab the first frame before polling starts.
        # Without this, video files trigger an immediate "video ended" on the first poll.
        self.gui.after(100, self._poll_frames)

    def _model_loop(self):
        model_fps_count = 0
        model_fps_t0 = time.time()

        while self.preview_active:
            frame0, frame1 = self._frame_queue.get()

            if not self._estimation_active:
                continue

            result = self.model.process(frame0, frame1)
            model_fps_count += 1
            elapsed = time.time() - model_fps_t0
            if elapsed >= 2.0:
                self._last_model_fps = model_fps_count / elapsed
                model_fps_count = 0
                model_fps_t0 = time.time()

            if result is not None:
                self.gui.after(0, lambda r=result: self._update_gui(r))

    def _update_gui(self, result):
        """Update GUI with model results. Called from GUI thread."""
        self.gui.update_3d_plot(result["landmarks_3d"])
        self.gui.update_joint_angles(result.get("joint_angles", {}))
        
    def _stop_preview(self):
        """Stop frame polling and release cameras."""
        self.preview_active = False
        if self.cam0:
            self.cam0.release()
            self.cam0 = None
        if self.cam1:
            self.cam1.release()
            self.cam1 = None
        log.info("Preview stopped")
        
    def on_preview_visibility(self, visible: bool):
        """Stop or resume sending frames to camera previews."""
        self._show_preview = visible
        
    def on_toggle_estimation(self):
        self._estimation_active = not self._estimation_active
        self.gui.update_estimation_state(self._estimation_active)

    def on_toggle_recording(self):
        if self._recording:
            self.stop_recording()
        else:
            self.start_recording()
        self.gui.update_recording_state(self._recording)
    
    def _poll_frames(self):
        """
        Read one frame from each camera, send to GUI preview and
        put into frame_queue for model processing. Reschedules itself.
        """
        if not self.preview_active:
            return

        t0 = time.perf_counter()
        ret0, frame0, t0_capture, ver0 = self.cam0.read()
        t1 = time.perf_counter()
        ret1, frame1, t1_capture, ver1 = self.cam1.read()
        t2 = time.perf_counter()
        sync_diff_ms = abs(t0_capture - t1_capture) * 1000
        
        self.perf.record(
            cam0_read_ms = (t1 - t0) * 1000,
            cam1_read_ms = (t2 - t1) * 1000,
            sync_diff_ms = sync_diff_ms,
        )
        
        if ver0 == self._last_frame_version0 and ver1 == self._last_frame_version1:
            self.gui.after(PREVIEW_INTERVAL_MS, self._poll_frames)
            return
        
        self._last_frame_version0 = ver0
        self._last_frame_version1 = ver1
        
        if frame0 is None or frame1 is None:
            self.gui.after(PREVIEW_INTERVAL_MS, self._poll_frames)
            return
        
        self._last_frame_version0 = ver0
        self._last_frame_version1 = ver1
        
        if frame0 is None or frame1 is None:
            self.gui.after(PREVIEW_INTERVAL_MS, self._poll_frames)
            return
        
        if self._recording and self._writer0 and self._writer1:
            f0 = frame0 if frame0.ndim == 3 else cv.cvtColor(frame0, cv.COLOR_GRAY2BGR)
            f1 = frame1 if frame1.ndim == 3 else cv.cvtColor(frame1, cv.COLOR_GRAY2BGR)
            self._writer0.write(f0)
            self._writer1.write(f1)
                
        if not ret0 or not ret1:
            if self.mode == "video":
                log.info("Video ended, looping")
                self.cam0.set(cv.CAP_PROP_POS_FRAMES, 0)
                self.cam1.set(cv.CAP_PROP_POS_FRAMES, 0)
            else:
                if not ret0: log.warning("Failed to read frame from cam0")
                if not ret1: log.warning("Failed to read frame from cam1")
            self.gui.after(PREVIEW_INTERVAL_MS, self._poll_frames)
            return

        # Poll FPS
        self._fps_count += 1
        elapsed = time.time() - self._fps_t0
        if elapsed >= 2.0:
            poll_fps = self._fps_count / elapsed
            self._poll_fps = poll_fps
            log.info(f"Poll FPS: {poll_fps:.1f}")
            self.perf.record(
                poll_fps=poll_fps,
                cam0_capture_fps=self.cam0.capture_fps,
                cam1_capture_fps=self.cam1.capture_fps,
            )
            self._fps_count = 0
            self._fps_t0 = time.time()

        try:
            self._frame_queue.put_nowait((frame0, frame1))
        except queue.Full:
            pass
        if self._show_preview:
            prev0, pver0 = self.cam0.read_preview()
            prev1, pver1 = self.cam1.read_preview()
            if prev0 is not None and pver0 != self._last_preview_version0:
                self.gui.update_cam0(prev0)
                self._last_preview_version0 = pver0
            if prev1 is not None and pver1 != self._last_preview_version1:
                self.gui.update_cam1(prev1)
                self._last_preview_version1 = pver1

        self.gui.after(PREVIEW_INTERVAL_MS, self._poll_frames)  
    
    def _start_gui(self):
        """Create and launch the main window."""
        from gui.gui import MainWindow
        self.gui = MainWindow(controller=self)
        if self.model.triangulator is not None:
            self.gui.show_calibration_status("Cameras calibrated")
        self._restore_camera_selection()
        self.gui.run()