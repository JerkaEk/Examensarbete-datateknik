from __future__ import annotations

import subprocess
import yaml
from typing import List, Dict

try:
  from picamera2 import Picamera2
  HAS_PICAMERA = True
except ImportError:
  HAS_PICAMERA = False

"""
Camera Detection

"""

def detect_csi_camera() -> List[Dict]:
  """
  Detects any cameras connected via CSI.
  Returns a list of dicts with camera info.
  """
  
  cameras = []
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
    print(f"[WARN] Could not detect any CSI cameras:{e}")
  return cameras

# Always skip internal PI devices. This list can be expanded in the future.
INTERNAL_DEVICES = ["pispbe", "rp1-cfe", "rpi-hevc-dec"]

def detect_usb_camera() -> List[Dict]:
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
    print(f"[WARN] Could not detect any USB cameras:{e}")
  return cameras

def detect_all_cameras() -> List[Dict]:
  """
  Detects all cameras using detect_csi_camera and detect_usb_camera.
  """
  print("\n[INFO] Scanning for cameras...")
  csi_cameras = detect_csi_camera()
  usb_cameras = detect_usb_camera()
  all_cameras = csi_cameras + usb_cameras
  
  if not all_cameras:
    print("[ERROR] No cameras found. Check connections and try again")
    return []
  
  print(f"[INFO] Found {len(all_cameras)} cameras:")
  for i, cam in enumerate(all_cameras):
    print(f" [{i}] {cam['display']}")
  
  return all_cameras

"""
User selection
"""

def select_cameras(cameras: List[Dict]) -> tuple:
  """
  If exactly 2 cameras are found, those gets selected automatically.
  If more, the user gets to pick which 2 to use.
  Useful for laptop cases.
  Returns (camera0, camera1).
  """
  
  if len(cameras) == 2:
    print(f"[INFO] Automatically selected:")
    print(f"  camera0: {cameras[0]['display']}")
    print(f"  camera1: {cameras[1]['display']}")
    return cameras[0], cameras[1]
  
  print("\n[INPUT] more than 2 cameras found. Please select which 2 to use")

  while True:
    try:
      idx0 = int(input(f"  Select camera0 (0-{len(cameras)-1}): "))
      idx1 = int(input(f"  Select camera1 (0-{len(cameras)-1}): "))
      if idx0 == idx1:
        print("[ERROR] camera0 and camera1 must be different.")
        continue
      if 0 <= idx0 < len(cameras) and 0 <= idx1 < len(cameras):
        return cameras[idx0], cameras[idx1]
      print(f"[ERROR] Please enter number between 0 and {len(cameras)-1}.")
    except ValueError:
      print("[ERROR] Please enter a valid number.")
      
"""
User Configuration
"""

def ask_int(prompt: str, default: int) -> int:
  """Ask user for an interger value with a default value if the user doesn't respond."""
  val = input(f"  {prompt} [default: {default}]: ").strip() # Strips off anything entered after the number (accidental spaces eg.)
  if val == "":
    return default
  try:
    return int(val)
  except ValueError:
    print(f"[WARN] Invalid input, using default: {default}")
    return default

def ask_float(prompt: str, default: float) -> float:
    """Ask user for a float value with a default value if the user doesn't respond."""
    val = input(f"  {prompt} [default: {default}]: ").strip()
    if val == "":
        return default
    try:
        return float(val)
    except ValueError:
        print(f"[WARN] Invalid input, using default: {default}")
        return default
      
def configure_settings() -> Dict:
  """Ask user for calibration settings, falling back to sensible defaults."""
  print("\n[INPUT] Configure calibration settings (press Enter to use defaults):")
 
  settings = {}
 
  settings["frame_width"] = ask_int("Frame width (px)", 640)
  settings["frame_height"] = ask_int("Frame height (px)", 480)
  settings["checkerboard_rows"] = ask_int("Checkerboard inner rows", 9)
  settings["checkerboard_columns"] = ask_int("Checkerboard inner columns", 6)
  settings["checkerboard_box_size_scale"] = ask_float("Checkerboard square size (cm)", 2.5)
  settings["mono_calibration_frames"] = ask_int("Frames to capture per camera", 15)
  settings["stereo_calibration_frames"] = ask_int("Stereo frame pairs to capture", 15)
  settings["view_resize"] = ask_int("Preview downscale factor", 2)
  settings["cooldown"] = ask_int("Cooldown frames between captures", 20)

  return settings
  
def generate_yaml(cam0: Dict, cam1: Dict, settings: Dict, output_path: str) -> None:
  """Generate and save calibration_settings.yaml."""

  # Camera IDs — int for CSI, string path for USB
  cam0_id = cam0["id"]
  cam1_id = cam1["id"]

  yaml_data = {
    "camera0": cam0_id,
    "camera1": cam1_id,
    "frame_width": settings["frame_width"],
    "frame_height": settings["frame_height"],
    "checkerboard_rows": settings["checkerboard_rows"],
    "checkerboard_columns": settings["checkerboard_columns"],
    "checkerboard_box_size_scale": settings["checkerboard_box_size_scale"],
    "mono_calibration_frames": settings["mono_calibration_frames"],
    "stereo_calibration_frames": settings["stereo_calibration_frames"],
    "view_resize": settings["view_resize"],
    "cooldown": settings["cooldown"],
  }

  with open(output_path, "w") as f:
    yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

  print(f"\n[INFO] Settings saved to '{output_path}'")
  print("[INFO] You can now run:")
  print(f"         python dynamic_cam_calibration.py {output_path}")


def main() -> None:
  print("=" * 50)
  print("  Camera Auto-Setup")
  print("=" * 50)       
        
  cameras = detect_all_cameras()
  cam0, cam1 = select_cameras(cameras)
  settings = configure_settings()
  
  output_path = input("\n  Output filename [default: calibration_settings.yaml]: ").strip()
  if output_path == "":
    output_path = "calibration_settings.yaml"

  generate_yaml(cam0, cam1, settings, output_path)      
  
if __name__ == "__main__":
  main()
  
      
    

  