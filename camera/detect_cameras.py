import subprocess
import logging

from typing import List, Dict

log = logging.getLogger(__name__)

try:
  from picamera2 import Picamera2
  HAS_PICAMERA = True
except ImportError:
  HAS_PICAMERA = False
  
# Always skip internal PI devices. This list can be expanded in the future.
INTERNAL_DEVICES = ["pispbe", "rp1-cfe", "rpi-hevc-dec"]

def detect_all_cameras() -> List[Dict]:
  """
  Detects all cameras using detect_csi_camera and detect_usb_camera.
  """
  log.info("Scanning for cameras...")
  csi_cameras = _detect_csi_camera()
  usb_cameras = _detect_usb_camera()
  all_cameras = csi_cameras + usb_cameras
  
  if not all_cameras:
    log.warning("No cameras found. Check connections and try again")
    return []
  
  log.info(f"Found {len(all_cameras)} cameras:")
  for i, cam in enumerate(all_cameras):
    log.info(f" [{i}] {cam['display']}")
  
  return all_cameras

def _detect_csi_camera() -> List[Dict]:
  """
  Detects any cameras connected via CSI.
  Returns a list of dicts with camera info.
  """
  cameras = []
  if not HAS_PICAMERA:
    return cameras
  try:
    # Find all connected cameras
    info = Picamera2.global_camera_info()
    for cam in info:
      # Skip USB cameras — their Id contains "usb"
      if "usb" in cam.get("Id", "").lower():
        continue
      cameras.append({
        "id": cam["Num"],
        "type": "CSI",
        "model": cam.get("Model", "unknown"),
        "display": f"CSI camera {cam['Num']} ({cam.get('Model', 'unknown')})"
      })
  except Exception as e:
    log.warning(f"Could not detect any CSI cameras:{e}")
  return cameras

def _detect_usb_camera() -> List[Dict]:
  """
  Detects any cameras connected via USB.
  Returns a list of dicts with camera info.
  """
  cameras = []
  try:
    res = subprocess.run( 
            ["v4l2-ctl", "--list-devices"], 
            capture_output=True, text=True
    )
    
    lines = res.stdout.splitlines() # split output into lines
    
    current_name = None
    for line in lines:
      if not line.startswith("\t") and line.strip():  # Lines starting with \t are device names
        current_name = line.strip().rstrip(":")
      elif line.strip().startswith("/dev/video"): # If device path
        device = line.strip()
        check = subprocess.run(
          ["v4l2-ctl", "--device", device, "--list-formats"], # Grabs output format from device
          capture_output=True, text=True
        )
        if "MJPG" in check.stdout or "YUYV" in check.stdout or "H264" in check.stdout: 
          if current_name and not any(d in current_name for d in INTERNAL_DEVICES):
            
            cameras.append({
              "id": device,
              "type": "USB",
              "model": current_name,
              "display": f"USB camera {device} ({current_name})"
            })
  except Exception as e:
    log.warning(f"Could not detect any USB cameras:{e}")
  return cameras