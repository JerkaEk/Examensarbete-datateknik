import logging
import time
import queue
import threading

from camera.camera_capture import open_camera
from core.model import Model
from utils.utils_io import load_yaml, save_yaml

#from calibration import calibration, auto_settings
#from pose_estimaiton import pose2d_extractor, triangulate3d #?

log = logging.getLogger(__name__)

PREVIEW_INTERVAL_MS = int(1000/30)

class Controller:
    """Main controller for the Stereo 3D Skeleton Tracker."""
    def __init__(self, no_gui=False, mock_cameras=False):   
        self.no_gui = no_gui
        self.mock_cameras = mock_cameras
        
        # State 
        self.preview_active = False
        self.gui = None
        self.calibration_running = False
        
        # Cameras
        self.cam0_id = None
        self.cam1_id = None
        self.cam0 = None
        self.cam1 = None

        # Model
        self.model = Model()
        
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
        success = run_calibration("calibration/calibration_settings.yaml")

        if success:
            log.info("Calibration complete")
        else:
            log.error("Calibration failed")

        self.calibration_running = False
        if self.cam0_id and self.cam1_id:
            self._start_preview()
        
    def on_calibration_settings_opened(self):
        """Called by GUI when calibration settings panel opens — loads yaml into GUI."""
        log.debug("Calibration settings opened")
        values = load_yaml("calibration_settings.yaml")
        self.gui.populate_settings(values)
        
    def on_calibration_settings_saved(self, values: dict):
        """Called by GUI when user saves calibration settings."""
        log.info("Saving calibration settings")
        from calibration.auto_settings import generate_yaml
        # TODO: generate_yaml needs cam0/cam1
        
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
    
    # Send to GUI
    
    # Utilities
    
    def on_camera_settings_saved(self, cam0: dict, cam1: dict):
        """
        Called by GUI when user saves camera selection.
        """
        log.info(f"Selected cameras: cam0={cam0['display']}, cam1={cam1['display']}")
        self.cam0_id = cam0['id']
        self.cam1_id = cam1['id']
        
        settings = load_yaml("calibration_settings.yaml")
        settings["camera0"] = cam0["id"]
        settings["camera1"] = cam1["id"]
        save_yaml(settings, "calibration_settings.yaml")
        
        self._start_preview()    
    
    def _start_preview(self):
        """Open cameras and begin polling frames."""
        log.info(f"Opening cameras: cam0={self.cam0_id}, cam1={self.cam1_id}")
        self.cam0 = open_camera(self.cam0_id)
        self.cam1 = open_camera(self.cam1_id)
        self.preview_active = True
        log.info("Preview Started")
        
        self._frame_queue = queue.Queue(maxsize=2)
        
        self._fps_count = 0
        self._fps_t0 = time.time()
        self._model_fps_count = 0
        self._model_fps_t0 = time.time()
        
        self._model_thread = threading.Thread(target=self._model_loop, daemon=True)
        self._model_thread.start()
        self._poll_frames()
    
    def _model_loop(self):
        """Run model in separate thread"""
        while self.preview_active:
            try:
                frame0, frame1 = self._frame_queue.get(timeout=1)
                result = self.model.process(frame0, frame1)
                self.gui.after(0, lambda r=result: self._update_gui(r))
            except queue.Empty:
                continue
                
    def _update_gui(self, result):
        """Update GUI with model results. Called from GUI thread."""
        self.gui.update_3d_plot(result["landmarks_3d"])
        
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
    
    def _poll_frames(self):
        """Read one frame from each camera and send to GUI. Reschedules itself."""
        if not self.preview_active:
            return

        ret0, frame0 = self.cam0.read()
        ret1, frame1 = self.cam1.read()

        # Poll FPS
        self._fps_count += 1
        elapsed = time.time() - self._fps_t0
        if elapsed >= 2.0:
            log.info(f"Poll FPS: {self._fps_count / elapsed:.1f}")
            self._fps_count = 0
            self._fps_t0 = time.time()

        if ret0 and ret1:
            try:
                self._frame_queue.put_nowait((frame0, frame1))
            except queue.Full:
                pass
            self.gui.update_cam0(frame0)
            self.gui.update_cam1(frame1)
        else:
            if not ret0:
                log.warning("Failed to read frame from cam0")
            if not ret1:
                log.warning("Failed to read frame from cam1")

        self.gui.after(PREVIEW_INTERVAL_MS, self._poll_frames)
    
    def _start_gui(self):
      """Create and launch the main window."""
      from gui.gui_new import MainWindow
      self.gui = MainWindow(controller=self)
      self.gui.run()