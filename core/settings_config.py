"""
Definition of all calibration settings within the app.
"""


CALIBRATION_SETTINGS = [
    {
        "key":     "frame_width",
        "label":   "Frame width",
        "default": 1280,
        "type":    int,
        "tooltip": "Camera frame width in pixels.",
    },
    {
        "key":     "frame_height",
        "label":   "Frame height",
        "default": 720,
        "type":    int,
        "tooltip": "Camera frame height in pixels.",
    },
    {
        "key":     "checkerboard_rows",
        "label":   "Checkerboard rows",
        "default": 6,
        "type":    int,
        "tooltip": "Number of inner corners along the rows of the checkerboard.",
    },
    {
        "key":     "checkerboard_columns",
        "label":   "Checkerboard columns",
        "default": 9,
        "type":    int,
        "tooltip": "Number of inner corners along the columns of the checkerboard.",
    },
    {
        "key":     "checkerboard_box_size_scale",
        "label":   "Checkerboard box size",
        "default": 2.5,
        "type":    float,
        "tooltip": "Physical size of each checkerboard square in centimeters.",
    },
    {
        "key":     "mono_calibration_frames",
        "label":   "Mono calibration frames",
        "default": 15,
        "type":    int,
        "tooltip": "Number of frames to capture per camera during mono calibration.",
    },
    {
        "key":     "stereo_calibration_frames",
        "label":   "Stereo calibration frames",
        "default": 15,
        "type":    int,
        "tooltip": "Number of frame pairs to capture during stereo calibration.",
    },
    {
        "key":     "view_resize",
        "label":   "View resize",
        "default": 1.0,
        "type":    float,
        "tooltip": "Scale factor for the preview window (1.0 = original size, 0.5 = half size).",
    },
    {
        "key":     "cooldown",
        "label":   "Cooldown",
        "default": 1.0,
        "type":    float,
        "tooltip": "Seconds to wait between each frame capture during calibration.",
    },
]