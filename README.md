# Stereo Camera 3D Skeleton Tracker

A real time 3D human pose estimation system built on a Raspberry Pi 5 using a stereo camera setup.
This was a part of a bachelor's thesis in computer engineering.

**Status:** Active development - feasibility study

## Overview
The system implements a complete pipeline that captures synchronized videos from two cameras, runs 2D estimation on each frame independently, then triangulates the results into 3D coordinates. A live GUI displays both camera views and an interactive 3D skeleton plot with joint angles.

## Hardware
- Raspberry Pi 5 8GB
- 2x InnoMaker OV9281
- 3D printed mount with a 7.5cm distance between cameras.

## Features
- Synchronized dual camera capture via `picamera2` (CSI) and OpenCV (USB).
- 2D pose estimation using **MediaPipe** with a simple option to swap to **MoveNet** (lightning/thunder) by changing two lines of code.
- Stereo triangulation with OpenCV.
- Interactive live skeleton visualization with toggleable joint angle readouts.
- Per component performance logging.
- Synchronized video recording and option to use pre recorded video instead of live estimation.

## Architecture
```
app.py  
├─calibration/
│   └─ calibration.py
├─camera/
│   └─ camera_capture.py
│   └─ detect_cameras.py
├─core/
│   └─ controller.py
│   └─ model.py
│   └─ performance_logger.py
├─gui/
│   └─ gui.py
├─pose_estimation/
│   └─ body_model.py
│   └─ mediapipe_estimator.py
│   └─ movenet_estimator.py
│   └─ triangulate.py
├─utils/
│   └─ utils_io.py
```
## Requirements
- Python 3.11
- Linux (Raspberry Pi OS / Ubuntu) or Windows

## Installation

### Linux
```bash
git clone https://github.com/JerkaEk/Examensarbete-datateknik.git
cd Examensarbete-datateknik
pip install -r requirements.txt --break-system-packages
```

### Windows
```bash
git clone https://github.com/JerkaEk/Examensarbete-datateknik.git
cd Examensarbete-datateknik
pip install -r requirements_win.txt
```

### Models
**MediaPipe** — download and place in `pose_estimation/`:
```bash
wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task -P pose_estimation/
```

**MoveNet** — downloaded automatically to `models/` on first run.

## Run
```bash
python app.py
```
> **Note:** Camera calibration is required before first use. See [Calibration](#calibration).

### MediaPipe landmarks models
Download and place in ```pose_estimation```:
```
wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task -P pose_estimation/
```
### MoveNet models
Downloaded automatically to ```models/``` on first use.

## Calibration
To calibrate the cameras a checkerboard or checkerboard pattern is required. One can be found and printed [here.](https://github.com/kyle-bersani/opencv-examples/blob/master/CalibrationByChessboard/chessboard-to-print.pdf)
Calibrate the cameras (intrinsic and extrinsic) by pressing **Calibrate** in the GUI.  
Camera parameters are saved to ```camera/camera_parameters```.  
The system loads them automatically on startup if they exist.

## Switching estimator
in ```core/model.py```, comment/uncomment the estimator import:
```
from pose_estimation.pose2d_extractor import Pose2DEstimator as _Estimator
_ESTIMATOR_KWARGS = {"model_complexity": 0}

# from pose_estimation.movenet_estimator import MoveNetEstimator as _Estimator
# _ESTIMATOR_KWARGS = {"variant": "lightning"}
```

## Landmark schema
Both MediaPipe (33 landmarks) and MoveNet (17 landmarks) are mapped to share a 13-landmark body model, this is defined in ```pose_estimation/body_model.py```

## Known limitations
- MediaPipe GPU delegate is not supported on Windows and Raspberry Pi 5 doesn't have a GPU for accelerated processing.
- ````matplotlib``` canvas redraws every frame regardless of detection.
- The system only supports estimation of one person in frame.

## Credits
This project is based on [Stereo-3D-Skeleton-Tracker](https://github.com/ranagursoy/Stereo-3D-Skeleton-Tracker) by ranagursoy.

## License
TBA
