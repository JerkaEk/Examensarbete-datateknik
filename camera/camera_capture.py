import cv2 as cv
import sys
import logging

log = logging.getLogger(__name__)

try:
    from picamera2 import Picamera2
    HAS_PICAMERA = True
except ImportError:
    HAS_PICAMERA = False

class PiCameraCapture:
    """Wrapper around Picamera2 to mimic cv.VideoCapture interface."""
    # This function is first called in open_camera() when a CSI camera is detected
    def __init__(self, camera_index: int, width: int = 640, height: int = 480):
        #initialize the camera with the given index and set the resolution
        self.cam = Picamera2(camera_index)
        config = self.cam.create_preview_configuration(
            main={"format": "RGB888", "size": (width, height)}
        )
        self.cam.configure(config)
        # start the camera to begin capturing frames
        self.cam.start()
    # The read() method captures a frame from the camera 
    def read(self):
        frame = self.cam.capture_array()
        # Convert from RGB to BGR format for OpenCV compatibility
        bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR) # stod XYZ istället för BGR, kolla detta
        return True, bgr
    # The release() method stops the camera and releases any resources
    def release(self):
        self.cam.stop()
        self.cam.close()
   # The isOpened() method always returns True since the camera is initialized in the constructor
    def isOpened(self):
        return True
    

# Dynamic camera selection. Uses default witdth/height if no parameters are provided
def open_camera(camera_id, width=1920, height=1080, fps=120):
  """
  Try cv.VideoCapture first (USB) with PiCameraCapture as fallback(CSI).
  Returns a camera object with read() and release() interface.
  """
  # Video file — don't apply resolution/fps settings
  if isinstance(camera_id, str) and camera_id.endswith((".mp4", ".avi", ".mkv")):
        cap = cv.VideoCapture(camera_id)
        if cap.isOpened():
            log.info(f"Opened video file: {camera_id}")
            return cap
        log.error(f"Could not open video file: {camera_id}")
        return None

  # Linux — USB camera
  if isinstance(camera_id, str) and camera_id.startswith("/dev/video"):
    cap = cv.VideoCapture(camera_id)
    if cap.isOpened():
      cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
      cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
      cap.set(cv.CAP_PROP_FPS, fps)
      actual_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
      actual_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
      actual_fps = cap.get(cv.CAP_PROP_FPS)
      log.info(f"Opened USB Camera {camera_id} at {actual_w}x{actual_h} {actual_fps:.0f}fps")
      return cap

  # Windows — USB camera
  if isinstance(camera_id, int) and not HAS_PICAMERA:
        cap = cv.VideoCapture(camera_id)
        if cap.isOpened():
            cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv.CAP_PROP_FPS, fps)
            actual_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv.CAP_PROP_FPS)
            log.info(f"Opened camera index {camera_id} at {actual_w}x{actual_h} {actual_fps:.0f}fps")
            return cap
        log.error(f"Could not open camera index {camera_id}")
        return None

  # Fallback — CSI camera via PiCameraCapture
  if isinstance(camera_id, int):
    if not HAS_PICAMERA:
        log.warning("picamera2 not available, cannot open CSI camera")
        return None
    try:
        cap = PiCameraCapture(camera_id, width=width, height=height)
        log.info(f"Opened CSI camera {camera_id} at {width}x{height}")
        return cap
    except Exception as e:
        log.error(f"PiCamera failed: {e}")
    sys.exit(f"[ERROR] could not open camera: {camera_id}")