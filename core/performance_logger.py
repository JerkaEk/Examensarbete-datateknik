import os
import time
import csv
import logging
import threading
import numpy as np

from collections import defaultdict
from utils.utils_io import save_csv

log = logging.getLogger(__name__)

class PerformanceLogger:
  def __init__(
    self,
    log_interval_s: float = 5.0,
    log_to_csv: bool = False,
    csv_path: str = None,
    ):
    self.log_interval_s = log_interval_s
    self.log_to_csv = log_to_csv

    if log_to_csv:
      if csv_path is None:
        os.makedirs("logs", exist_ok=True)
        n = 1
        while os.path.exists(f"logs/performance_log_{n}.csv"):
          n += 1
        csv_path = f"logs/performance_log_{n}.csv"
      self.csv_path = csv_path

    self._buffer = defaultdict(list)
    self._last_log_t = time.perf_counter()
    self._rows = []

    # Guards _buffer and _rows. record() is called from multiple threads
    # (GUI poll loop + model loop) and without this we get
    # "dictionary changed size during iteration" in _log_summary.
    self._lock = threading.Lock()

  def record(self, **kwargs):
    """
    Record a set of timing values. Thread-safe.
    """
    with self._lock:
      for key, value in kwargs.items():
        self._buffer[key].append(value)
      if self.log_to_csv:
        self._rows.append({"timestamp": time.perf_counter(), **kwargs})

      due = time.perf_counter() - self._last_log_t >= self.log_interval_s
      if due:
        self._log_summary_locked()
        self._last_log_t = time.perf_counter()

  def _log_summary_locked(self):
    """Caller must hold self._lock."""
    log.info("=== Performance Summary ===")
    # Snapshot the keys so we can iterate safely even if logic below
    # ever mutates the buffer.
    for key in list(self._buffer.keys()):
      values = self._buffer[key]
      if not values:
        continue
      arr = np.array(values)
      mean = np.mean(arr)
      std = np.std(arr)
      if key.endswith("_fps"):
        log.info(f"  {key:<26} mean={mean:7.1f} fps  std={std:6.1f}")
      else:
        fps = 1000 / mean if mean > 0 else 0
        log.info(f"  {key:<26} mean={mean:7.1f} ms   std={std:6.1f} ms   fps={fps:6.1f}")
    if self._buffer:
      log.info(f"  samples: {max(len(v) for v in self._buffer.values())}")
    self._buffer.clear()

  def close(self):
    with self._lock:
      if self._buffer:
        self._log_summary_locked()
      if self.log_to_csv and self._rows:
        rows_to_save = self._rows
        path = self.csv_path
        self._rows = []
      else:
        rows_to_save = None
        path = None

    # save_csv outside the lock — no shared state needed and it can be slow.
    if rows_to_save:
      save_csv(rows_to_save, path)
      log.info(f"Performance log saved to {path}")