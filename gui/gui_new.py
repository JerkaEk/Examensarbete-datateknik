import logging

import customtkinter as ctk
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D # noqa: F401

from core.settings_config import CALIBRATION_SETTINGS

log = logging.getLogger(__name__)

# -------------------------------------------------- #
# Themes & Size Settings
# -------------------------------------------------- #
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

SETTINGS_PANEL_WIDTH = 320
SETTINGS_PANEL_SLIDE_STEPS = 12
SETTINGS_PANEL_DELAY=10
# -------------------------------------------------- #
# Main Window
# -------------------------------------------------- #
class MainWindow(ctk.CTk):
  def __init__(self, controller):
    super().__init__()
    self.controller = controller
    
    self.title("Stereo 3D Skeleton Tracker")
    self.geometry("1600x900")
    #self.geometry("1280x720")
    self.minsize(800,500)
    
    self._camera_settings_panel_open = False
    self._calibration_settings_panel_open = False
    
    self._camera_settings_built = False
    self._calibration_settings_built = False
    
    self._available_cameras = []
    
    self._build_topbar()
    self._build_content()
    
  # -------------------------------------------------- #
  # Build Components
  # -------------------------------------------------- #
  def _build_topbar(self):
    self.topbar = ctk.CTkFrame(self, height=48, corner_radius=0)
    self.topbar.pack(side="top", fill="x")
    self.topbar.pack_propagate(False)

    self.btn_camera_settings = ctk.CTkButton(
        self.topbar,
        text="Camera settings",
        command=self._on_camera_settings,
    )
    self.btn_camera_settings.pack(side="left", padx=10, pady=8)

    self.btn_calibration_settings = ctk.CTkButton(
        self.topbar,
        text="Calibration settings",
        command=self._on_calibration_settings,
    )
    self.btn_calibration_settings.pack(side="left", padx=10, pady=8)

    self.btn_calibrate = ctk.CTkButton(
        self.topbar,
        text="Calibrate",
        command=self._on_calibrate,
    )
    self.btn_calibrate.pack(side="left", padx=10, pady=8)
    
  def _build_content(self):
    # Outer container under top bar
    self.content = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
    self.content.pack(side="top", fill="both", expand=True)
    
    # Camera settings panel
    self.camera_settings_panel = ctk.CTkFrame(self.content, width=0, corner_radius=0)
    self.camera_settings_panel.pack(side="left", fill="y")
    self.camera_settings_panel.pack_propagate(False)
    
    # Calibration settings panel
    self.calibration_settings_panel = ctk.CTkFrame(self.content, width=0, corner_radius=0)
    self.calibration_settings_panel.pack(side="left", fill="y")
    self.calibration_settings_panel.pack_propagate(False)
    
    # Config Columns - Camera previews
    self.camera_column = ctk.CTkFrame(self.content, corner_radius=0, fg_color="transparent", width=600)
    self.camera_column.pack(side="left", fill="y", padx=(8, 4), pady=8)
    self.camera_column.pack_propagate(False)
    
    # Config Columns - 3D-plot
    self.main = ctk.CTkFrame(self.content, corner_radius=0, fg_color="transparent")
    self.main.pack(side="left", fill="both", expand=True, padx=(4, 8), pady=8)
    self.main.rowconfigure(0, weight=1)
    self.main.columnconfigure(0, weight=1)
    
    self._build_plot_area()
    self._build_camera_preview()
    
  def _build_plot_area(self):
    self.plot_frame = ctk.CTkFrame(self.main)
    self.plot_frame.grid(row=0, column=0, sticky="nsew")

    # Matplotlib figure med mörk bakgrund
    self.fig = plt.Figure(facecolor="#2b2b2b")
    self.ax = self.fig.add_subplot(111, projection="3d")
    self._style_3d_axes()

    self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
    self.canvas.draw()
    self.canvas.get_tk_widget().pack(fill="both", expand=True)
    
    # Reset plot orientation
    self.btn_reset_view = ctk.CTkButton(
        self.plot_frame,
        text="Reset view",
        width=100,
        command=self._on_reset_view,
    )
    self.btn_reset_view.place(relx=0.0, rely=1.0, anchor="sw", x=8, y=-8)
    
  def _build_camera_preview(self):
    # Cam 0
    self.cam0_frame = ctk.CTkFrame(self.camera_column)
    self.cam0_frame.pack(side="top", fill="both", expand=True, pady=(0, 4))
    
    self.cam0_label = ctk.CTkLabel(
      self.cam0_frame,
      text="No signal – cam 0",
      text_color="gray50",
      font=ctk.CTkFont(size=14),
    )
    
    self.cam0_label.place(relx=0.5, rely=0.5, anchor="center")

    # Cam 1
    self.cam1_frame = ctk.CTkFrame(self.camera_column)
    self.cam1_frame.pack(side="top", fill="both", expand=True, pady=(4, 0))

    self.cam1_label = ctk.CTkLabel(
      self.cam1_frame,
      text="No signal – cam 1",
      text_color="gray50",
      font=ctk.CTkFont(size=14),
    )
    self.cam1_label.place(relx=0.5, rely=0.5, anchor="center")
    
  # -------------------------------------------------- #
  # Build Settings Panels
  # -------------------------------------------------- #  
  def _build_calibration_settings_panel(self):
    # Title
    ctk.CTkLabel(
      self.calibration_settings_panel,
      text="Calibration settings",
      font=ctk.CTkFont(size=15, weight="bold"),
    ).pack(pady=(16, 8), padx=16, anchor="w")

    ctk.CTkFrame(self.calibration_settings_panel, height=1, fg_color="gray30").pack(
      fill="x", padx=16, pady=(0, 12)
    )

    # Scrollable if window is too small
    self.settings_scroll = ctk.CTkScrollableFrame(
      self.calibration_settings_panel, fg_color="transparent"
    )
    self.settings_scroll.pack(fill="both", expand=True, padx=8)

    # One field per setting
    self._settings_fields = {}
    for setting in CALIBRATION_SETTINGS:
      self._add_setting_row(
        label=setting["label"],
        key=setting["key"],
        default=str(setting["default"]),
        tooltip=setting["tooltip"],
      )

    ctk.CTkFrame(self.calibration_settings_panel, height=1, fg_color="gray30").pack(
      fill="x", padx=16, pady=(8, 0)
    )

    btn_row = ctk.CTkFrame(self.calibration_settings_panel, fg_color="transparent")
    btn_row.pack(fill="x", padx=16, pady=12)

    ctk.CTkButton(
      btn_row,
      text="Save",
      command=self._on_calibration_settings_save,
    ).pack(side="left", expand=True, fill="x", padx=(0, 4))

    ctk.CTkButton(
      btn_row,
      text="Cancel",
      fg_color="gray30",
      hover_color="gray40",
      command=self._on_settings_calibration_cancel,
    ).pack(side="left", expand=True, fill="x", padx=(4, 0))
    
  def _build_camera_settings_panel(self):
    ctk.CTkLabel(
      self.camera_settings_panel,
      text="Camera settings",
      font=ctk.CTkFont(size=15, weight="bold"),
    ).pack(pady=(16, 8), padx=16, anchor="w")

    ctk.CTkFrame(self.camera_settings_panel, height=1, fg_color="gray30").pack(
      fill="x", padx=16, pady=(0, 12)
    )

    # Scan button
    self.btn_scan = ctk.CTkButton(
      self.camera_settings_panel,
      text="Scan for cameras",
      command=self._on_scan_cameras,
    )
    self.btn_scan.pack(padx=16, pady=(0, 8), fill="x")

    # Status label
    self.scan_status = ctk.CTkLabel(
      self.camera_settings_panel,
      text="Press scan to detect cameras",
      text_color="gray50",
      font=ctk.CTkFont(size=12),
      wraplength=SETTINGS_PANEL_WIDTH - 40,
    )
    self.scan_status.pack(padx=16, anchor="w")

    ctk.CTkFrame(self.camera_settings_panel, height=1, fg_color="gray30").pack(
      fill="x", padx=16, pady=12
    )

    # Cam 0 dropdown
    ctk.CTkLabel(
      self.camera_settings_panel,
      text="Camera 0",
      font=ctk.CTkFont(size=12),
      anchor="w",
    ).pack(padx=16, anchor="w")

    self.cam0_var = ctk.StringVar(value="No cameras detected")
    self.cam0_dropdown = ctk.CTkOptionMenu(
      self.camera_settings_panel,
      variable=self.cam0_var,
      values=["No cameras detected"],
      state="disabled",
    )
    self.cam0_dropdown.pack(padx=16, pady=(4, 12), fill="x")

    # Cam 1 dropdown
    ctk.CTkLabel(
      self.camera_settings_panel,
      text="Camera 1",
      font=ctk.CTkFont(size=12),
      anchor="w",
    ).pack(padx=16, anchor="w")

    self.cam1_var = ctk.StringVar(value="No cameras detected")
    self.cam1_dropdown = ctk.CTkOptionMenu(
      self.camera_settings_panel,
      variable=self.cam1_var,
      values=["No cameras detected"],
      state="disabled",
    )
    self.cam1_dropdown.pack(padx=16, pady=(4, 0), fill="x")

    # Save / Cancel
    ctk.CTkFrame(self.camera_settings_panel, height=1, fg_color="gray30").pack(
      fill="x", padx=16, pady=(16, 0)
    )

    btn_row = ctk.CTkFrame(self.camera_settings_panel, fg_color="transparent")
    btn_row.pack(fill="x", padx=16, pady=12)

    ctk.CTkButton(
        btn_row,
        text="Save",
        command=self._on_camera_save,
    ).pack(side="left", expand=True, fill="x", padx=(0, 4))

    ctk.CTkButton(
        btn_row,
        text="Cancel",
        fg_color="gray30",
        hover_color="gray40",
        command=self._on_camera_cancel,
    ).pack(side="left", expand=True, fill="x", padx=(4, 0))
      
  # -------------------------------------------------- #
  # Utility for Settings Panels
  # -------------------------------------------------- #
  def _add_setting_row(self, label: str, key: str, default: str, tooltip: str):
    row = ctk.CTkFrame(self.settings_scroll, fg_color="transparent")
    row.pack(fill="x", pady=4)

    label_row = ctk.CTkFrame(row, fg_color="transparent")
    label_row.pack(fill="x")

    ctk.CTkLabel(
      label_row,
      text=label,
      font=ctk.CTkFont(size=12),
      anchor="w",
    ).pack(side="left")

    help_btn = ctk.CTkLabel(
      label_row,
      text=" ?",
      font=ctk.CTkFont(size=12),
      text_color="gray50",
      cursor="question_arrow",
    )
    help_btn.pack(side="left")
    self._bind_tooltip(help_btn, tooltip)

    entry = ctk.CTkEntry(row, placeholder_text=default)
    entry.pack(fill="x", pady=(2, 0))
    entry.insert(0, default)

    self._settings_fields[key] = entry

  def _bind_tooltip(self, widget, text: str):
    tooltip_win = None

    def on_enter(event):
      nonlocal tooltip_win
      
      # Destroy old tooltip if already exists
      if tooltip_win is not None:
        try:
          tooltip_win.destroy()
        except Exception:
          pass
      
      x = widget.winfo_rootx() + 20
      y = widget.winfo_rooty() + 20
      tooltip_win = ctk.CTkToplevel(self)
      tooltip_win.wm_overrideredirect(True)
      tooltip_win.geometry(f"+{x}+{y}")
      ctk.CTkLabel(
        tooltip_win,
        text=text,
        fg_color="#1a1a2e",
        corner_radius=6,
        wraplength=220,
        justify="left",
        font=ctk.CTkFont(size=11),
      ).pack(padx=8, pady=6)

    def on_leave(event):
      nonlocal tooltip_win
      if tooltip_win:
        tooltip_win.destroy()
        tooltip_win = None

    widget.bind("<Enter>", on_enter)
    widget.bind("<Leave>", on_leave)
    
  def _on_scan_cameras(self):
    """Detect cameras and populate dropdowns."""
    self.scan_status.configure(text="Scanning...", text_color="gray50")
    self.update()

    # Import here to avoid issues on Windows
    from calibration.auto_settings import detect_all_cameras
    cameras = detect_all_cameras()

    if not cameras:
      self.scan_status.configure(
        text="No cameras found. Check connections.",
        text_color="red"
      )
      return

    self._available_cameras = cameras
    display_names = [c["display"] for c in cameras]

    self.cam0_dropdown.configure(values=display_names, state="normal")
    self.cam1_dropdown.configure(values=display_names, state="normal")

    # Auto-select if exactly 2 found
    self.cam0_var.set(display_names[0])
    self.cam1_var.set(display_names[1] if len(display_names) > 1 else display_names[0])

    self.scan_status.configure(
      text=f"Found {len(cameras)} camera(s).",
      text_color="green"
    )
  
  def _on_camera_save(self):
    """Save selected cameras to yaml via controller."""
    if not self._available_cameras:
      self.scan_status.configure(
        text="Scan for cameras first.",
        text_color="red"
      )
      return

    # Find selected camera dicts by display name
    cam0_display = self.cam0_var.get()
    cam1_display = self.cam1_var.get()

    cam0 = next((c for c in self._available_cameras if c["display"] == cam0_display), None)
    cam1 = next((c for c in self._available_cameras if c["display"] == cam1_display), None)

    if cam0 is None or cam1 is None:
      self.scan_status.configure(text="Invalid selection.", text_color="red")
      return

    if cam0["id"] == cam1["id"]:
      self.scan_status.configure(
        text="Camera 0 and Camera 1 must be different.",
        text_color="red"
      )
      return

    self.controller.on_camera_settings_saved(cam0, cam1)
    self._camera_settings_panel_open = False
    self._set_panel(self.camera_settings_panel, False)

  def _on_camera_cancel(self):
    self._camera_settings_panel_open = False
    self._set_panel(self.camera_settings_panel, False)
    
  
  # -------------------------------------------------- #
  # Panel Control
  # -------------------------------------------------- #
  def _set_panel(self, panel, open: bool):
    target = SETTINGS_PANEL_WIDTH if open else 0
    self.canvas.get_tk_widget().pack_forget()  # Hide canvas
    panel.configure(width=target)
    self.update()
    self.canvas.get_tk_widget().pack(fill="both", expand=True)  # Show canvas
  
  def _on_camera_settings(self):
    log.debug("Camera settings panel toggled")
    self._camera_settings_panel_open = not self._camera_settings_panel_open
    if self._camera_settings_panel_open and not self._camera_settings_built:
      self._build_camera_settings_panel()
      self._camera_settings_built = True
    self._set_panel(self.camera_settings_panel, self._camera_settings_panel_open)
    
  def _on_calibration_settings(self):
    log.debug("Calibration settings panel toggled")
    self._calibration_settings_panel_open = not self._calibration_settings_panel_open
    if self._calibration_settings_panel_open and not self._calibration_settings_built:
      self._build_calibration_settings_panel()
      self._calibration_settings_built = True
    self._set_panel(self.calibration_settings_panel, self._calibration_settings_panel_open)
    
  # -------------------------------------------------- #
  # Button Events
  # -------------------------------------------------- #
  
  def _on_calibrate(self):
    log.info("Calibrate clicked")
    self.controller.on_calibrate_clicked() 
    
  def _on_reset_view(self):
    self.ax.view_init(elev=20, azim=-60)  # matplotlib default
    self.canvas.draw()
    
  def _on_calibration_settings_save(self):
    values = {key: entry.get() for key, entry in self._settings_fields.items()}  
    log.info(f"Saving calibrations settings: {values}")
    self._calibration_settings_panel_open = False
    self._set_panel(self.calibration_settings_panel, False)

  def _on_settings_calibration_cancel(self):
    self._calibration_settings_panel_open = False
    self._set_panel(self.calibration_settings_panel, False)
  
  # -------------------------------------------------- # 
  # Public API
  # -------------------------------------------------- #
  def update_cam0(self, frame):
    """Update camera 0 preview with a new BGR frame."""
    h = self.cam0_frame.winfo_height()
    w = self.cam0_frame.winfo_width()
    if h < 2 or w < 2:
       return
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    ctk_img = ctk.CTkImage(Image.fromarray(rgb), size=(w, h))
    self.cam0_label.configure(image=ctk_img, text="")
    self.cam0_label.image = ctk_img
    
  def update_cam1(self, frame):
    """Update camera 1 preview with a new BGR frame."""
    h = self.cam1_frame.winfo_height()
    w = self.cam1_frame.winfo_width()
    if h < 2 or w < 2:
      return
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    ctk_img = ctk.CTkImage(Image.fromarray(rgb), size=(w, h))
    self.cam1_label.configure(image=ctk_img, text="")
    self.cam1_label.image = ctk_img

    
  def update_3d_plot(self, landmarks_3d: np.ndarray):
    """Update the 3D plot with new landmark coordinates."""
    valid = np.isfinite(landmarks_3d).all(axis=1)
    log.debug(f"Valid landmarks: {valid.sum()}/33")
    if valid.sum() > 0:
        log.debug(f"landmarks_3d min/max: {landmarks_3d[valid].min():.1f} /{landmarks_3d[valid].max():.1f}")

    CONNECTIONS = [
        (0, 11), (0, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (11, 23), (12, 24),
        (23, 25), (25, 27),
        (24, 26), (26, 28),
        (23, 24), (11, 12),
    ]

    self.ax.clear()
    self._style_3d_axes()

    if not np.isnan(landmarks_3d).all():
        x, y, z = landmarks_3d[:, 0], landmarks_3d[:, 1], landmarks_3d[:, 2]
        self.ax.scatter(x, y, z, c="blue", s=20)
        for i, j in CONNECTIONS:
            if np.isfinite(landmarks_3d[i]).all() and np.isfinite(landmarks_3d[j]).all():
                self.ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], c="red", linewidth=1.5)

    self.canvas.draw()
    
  def populate_settings(self, values: dict):
    """Populate calibration settings fields with values from a dict."""
    for setting in CALIBRATION_SETTINGS:
      key = setting["key"]
      if key in self._settings_fields:
        value = values.get(key, setting["default"])
        entry = self._settings_fields[key]
        entry.delete(0, "end")
        entry.insert(0, str(value))
        
  def show_calibration_status(self, message: str):
    """Display a calibration status message in the GUI."""
    log.info(f"Calibration status: {message}")
    # TODO: visa i GUI
 
  def show_error(self, message: str):
    """Display an error message in the GUI."""  
    log.error(f"Error: {message}")
    # TODO: visa i GUI
  
  # -------------------------------------------------- #
  # Styling
  # -------------------------------------------------- #
  def _style_3d_axes(self):
    self.ax.set_facecolor("#2b2b2b")
    self.ax.xaxis.pane.fill = False
    self.ax.yaxis.pane.fill = False
    self.ax.zaxis.pane.fill = False
    self.ax.xaxis.pane.set_edgecolor("#444444")
    self.ax.yaxis.pane.set_edgecolor("#444444")
    self.ax.zaxis.pane.set_edgecolor("#444444")
    self.ax.tick_params(colors="gray")
    self.ax.xaxis.label.set_color("gray")
    self.ax.yaxis.label.set_color("gray")
    self.ax.zaxis.label.set_color("gray")
    self.ax.set_xlabel("X")
    self.ax.set_ylabel("Y")
    self.ax.set_zlabel("Z")
    
  def run(self):
    self.mainloop()
    
if __name__ == "__main__":
  window = MainWindow(controller=None)
  window.run()
