import logging
from camera.camera_capture import open_camera
from calibration.auto_settings import generate_yaml, load_yaml

#from calibration import calibration, auto_settings
#from pose_estimaiton import pose2d_extractor, triangulate3d #?

log = logging.getLogger(__name__)

PREVIEW_INTERVAL_MS = int(1000/30)

class Controller:
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
  
  def run(self):
    if not self.no_gui:
      self._start_gui()
    else:
      self._run_headless()
  
  def shutdown(self):
    if self.preview_active:
      self._stop_preview()
      
    # Close all process threads here.    
      

  def on_calibrate_clicked(self):
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
      values = load_yaml("calibration_settings.yaml")
      self.gui.populate_settings(values)
      
    def on_calibration_settings_saved(self, values: dict):
      from calibration.auto_settings import generate_yaml
      # TODO: generate_yaml needs cam0/cam1
      
    
  def on_toggle_preview(self):
    if self.preview_active:
      log.info("Stopping preview")
      self._stop_preview()
    else:
      log.info("Starting preview")
      self._start_preview()
      
  
  # Send to GUI
  
  # Utilities
  
  def on_camera_settings_saved(self, cam0: dict, cam1: dict):
    """
    Called by GUI when user saves camera selection.
    """
    log.info(f"Selected cameras: cam0={cam0['display']}, cam1={cam1['display']}")
    self.cam0_id = cam0['id']
    self.cam1_id = cam1['id']
    self._start_preview()    
    
  def _start_preview(self):
    self.cam0 = open_camera(self.cam0_id)
    self.cam1 = open_camera(self.cam1_id)
    self.preview_active = True
    log.info("Preview Started")
    self._poll_frames()
    
  def _stop_preview(self):
    self.preview_active = False
    if self.cam0:
      self.cam0.release()
      self.cam0 = None
    if self.cam1:
      self.cam1.release()
      self.cam1 = None
    log.info("Preview stopped")
    
  def _poll_frames(self):
    if not self.preview_active:
      return
    
    ret0, frame0 = self.cam0.read()
    ret1, frame1 = self.cam1.read()

    if ret0:
      self.gui.update_cam0(frame0)
    if ret1:
      self.gui.update_cam1(frame1)

    self.gui.after(PREVIEW_INTERVAL_MS, self._poll_frames)
    
  def _start_gui(self):
    from gui.gui_new import MainWindow
    self.gui = MainWindow(controller=self)
    self.gui.run()
      
  
