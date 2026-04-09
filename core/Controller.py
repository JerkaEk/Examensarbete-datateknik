import logging

#from calibration import calibration, auto_settings
#from pose_estimaiton import pose2d_extractor, triangulate3d #?

log = logging.getLogger(__name__)

class Controller:
  def __init__(self, no_gui=False, mock_cameras=False):
      
    self.no_gui = no_gui
    self.mock_cameras = mock_cameras
    
    # State 
    self.preview_active = False
    self.gui = None
    self.calibration_running = False
  
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

    log.info("Starting calibration")
    self.calibration_running = True

    from calibration.calibration import run_calibration
    success = run_calibration("calibration/calibration_settings.yaml")

    if success:
        log.info("Calibration complete")
    else:
        log.error("Calibration failed")

    self.calibration_running = False
    
  def on_calibration_settings_clicked(self):
    log.info("Opening calibration settings")
    # TODO read calibration_settings.yaml.
    # TODO save if user makes changes.
    
    
  def on_toggle_preview(self):
    if self.preview_active:
      log.info("Stopping preview")
      self._stop_preview()
    else:
      log.info("Starting preview")
      self._start_preview()
      
  
  # Send to GUI
  
  # Utilities
    
  def _start_preview(self):
    # TODO implement into gui
    pass
    self.preview_active = True
    log.info("Preview started (stub)")
    
  def _stop_preview(self):
    # TODO Stop the preview
    
    self.preview_active = False
    log.info("Preview stopped (stub)")
    
  def _start_gui(self):
    from gui.gui_new import MainWindow
    self.gui = MainWindow(controller=self)
    self.gui.run()
      
  
