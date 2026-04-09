import customtkinter as ctk
import matplotlib as plt
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D # noqa: F401

from core.settings_config import CALIBRATION_SETTINGS

# Themes
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

SETTINGS_PANEL_WIDTH = 320
SETTINGS_PANEL_SLIDE_STEPS = 12
SETTINGS_PANEL_DELAY=10

class MainWindow(ctk.CTk):
  def __init__(self, controller):
    super().__init__()
    self.controller = controller
    
    self.title("Stereo 3D Skeleton Tracker")
    self.geometry("1600x900")
    #self.geometry("1280x720")
    self.minsize(800,500)
    
    self._panel_open = False
    self._panel_current_width = 0
    
    self._build_topbar()
    self._build_content()
    
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
    self.content.pack(side="top", fill="both", expand=True, padx=8, pady=8)
    
    self.settings_panel = ctk.CTkFrame(self.content, width=0, corner_radius=0)
    self.settings_panel.pack(side="left", fill="y")
    self.settings_panel.pack_propagate(False)

    self._build_calibration_settings_panel()
    
    # Config rows: 3D-plotting 2x size of camera preview
    self.main = ctk.CTkFrame(self.content, corner_radius=0, fg_color="transparent")
    self.main.pack(side="left", fill="both", expand=True, padx=8, pady=8)
    
    self.main.rowconfigure(0, weight=2)   # 3D-plot
    self.main.rowconfigure(1, weight=1)   # Cameras
    self.main.columnconfigure(0, weight=1)

    self.main.rowconfigure(0, weight=2)
    self.main.rowconfigure(1, weight=1)
    self.main.columnconfigure(0, weight=1)
    
    self._build_plot_area()
    self._build_camera_preview()
    
  def _build_plot_area(self):
    self.plot_frame = ctk.CTkFrame(self.main)
    self.plot_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))

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
    self.camera_row = ctk.CTkFrame(self.main, fg_color="transparent")
    self.camera_row.grid(row=1, column=0, sticky="nsew")

    self.camera_row.columnconfigure(0, weight=1)
    self.camera_row.columnconfigure(1, weight=1)
    self.camera_row.rowconfigure(0, weight=1)

    # Cam 0
    self.cam0_frame = ctk.CTkFrame(self.camera_row)
    self.cam0_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

    self.cam0_label = ctk.CTkLabel(
        self.cam0_frame,
        text="No signal – cam 0",
        text_color="gray50",
        font=ctk.CTkFont(size=14),
    )
    self.cam0_label.place(relx=0.5, rely=0.5, anchor="center")

    # Cam 1
    self.cam1_frame = ctk.CTkFrame(self.camera_row)
    self.cam1_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

    self.cam1_label = ctk.CTkLabel(
        self.cam1_frame,
        text="No signal – cam 1",
        text_color="gray50",
        font=ctk.CTkFont(size=14),
    )
    self.cam1_label.place(relx=0.5, rely=0.5, anchor="center")
    
  def _build_calibration_settings_panel(self):
    # Title
    ctk.CTkLabel(
      self.settings_panel,
      text="Calibration settings",
      font=ctk.CTkFont(size=15, weight="bold"),
    ).pack(pady=(16, 8), padx=16, anchor="w")

    ctk.CTkFrame(self.settings_panel, height=1, fg_color="gray30").pack(
      fill="x", padx=16, pady=(0, 12)
    )

    # Scrollable if window is too small
    self.settings_scroll = ctk.CTkScrollableFrame(
      self.settings_panel, fg_color="transparent"
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

    ctk.CTkFrame(self.settings_panel, height=1, fg_color="gray30").pack(
      fill="x", padx=16, pady=(8, 0)
    )

    btn_row = ctk.CTkFrame(self.settings_panel, fg_color="transparent")
    btn_row.pack(fill="x", padx=16, pady=12)

    ctk.CTkButton(
      btn_row,
      text="Save",
      command=self._on_settings_save,
    ).pack(side="left", expand=True, fill="x", padx=(0, 4))

    ctk.CTkButton(
      btn_row,
      text="Cancel",
      fg_color="gray30",
      hover_color="gray40",
      command=self._on_settings_cancel,
    ).pack(side="left", expand=True, fill="x", padx=(4, 0))
 
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

  # Button Events
  def _on_camera_settings(self):
    print("Camera settings clicked")
    # TODO: implement
    
  def _on_calibration_settings(self):
    print("Calibration settings clicked")
    self._panel_open = not self._panel_open
    self._slide_panel(opening=self._panel_open)
  
  def _on_calibrate(self):
    print ("Calibrate clicked")
    self.controller.on_calibrate_clicked() 
    
  def _on_reset_view(self):
    self.ax.view_init(elev=20, azim=-60)  # matplotlib default
    self.canvas.draw()
    
  def _on_settings_save(self):
    values = {key: entry.get() for key, entry in self._settings_fields.items()}
    print("Sparar inställningar:", values)
    # TODO: skicka till controller → generate_yaml()
    self._panel_open = False
    self._slide_panel(opening=False)

  def _on_settings_cancel(self):
    self._panel_open = False
    self._slide_panel(opening=False)
    
  # Public API
  def update_cam0(self, image):
    self.cam0_label.configure(image=image, text="")
    
  def update_cam1(self, image):
    self.cam1_label.configure(image=image, text="")
    
  def populate_settings(self, values: dict):
    for setting in CALIBRATION_SETTINGS:
      key = setting["key"]
      if key in self._settings_fields:
        value = values.get(key, setting["default"])
        entry = self._settings_fields[key]
        entry.delete(0, "end")
        entry.insert(0, str(value))
        
  def show_calibration_status(self, message: str):
    print(f"[STATUS] {message}")
    # TODO: visa i GUI
 
  def show_error(self, message: str):
    print(f"[ERROR] {message}")
    # TODO: visa i GUI
  
  # Styling
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
    
  def _slide_panel(self, opening: bool):
    target = SETTINGS_PANEL_WIDTH if opening else 0
    self._panel_current_width = float(target)
    self.settings_panel.configure(width=target)
    
  def run(self):
    self.mainloop()
    
if __name__ == "__main__":
  window = MainWindow()
  window.run()