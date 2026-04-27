import time
import csv
import logging
import numpy as np

from collections import defaultdict
from utils.utils_io import save_csv

log = logging.getLogger(__name__)

class PerformanceLogger:
  def __init__(
    self,
    log_interval_s: float = 5.0,
    log_to_csv: bool = False,
    csv_path: str = "performance_log.csv"
    ):
    self.log_interval_s = log_interval_s
    self.log_to_csv = log_to_csv
    self.csv_path = csv_path
    
    self._buffer = defaultdict(list)
    self._last_log_t = time.perf_counter()
    
    self._rows = []
    
    if log_to_csv:
      self._csv_file = open(csv_path, "w", newline="")
      self._csv_writer = csv.writer(self._csv_file)
      self._header_written = False  # Header only at first record()
      
  def record(self, **kwargs):
    """
    Record a set of timing values.
    """
    for key, value in kwargs.items():
      self._buffer[key].append(value)
    if self.log_to_csv:
      self._rows.append({"timestamp": time.perf_counter(), **kwargs})
    
    if time.perf_counter() - self._last_log_t >= self.log_interval_s:
      self._log_summary()
      self._last_log_t = time.perf_counter()
      
  def _log_summary(self):
    """
    Log mean/std/fps for all recorded metrics.
    """
    log.info("=== Performance Summary ===")
    for key, values in self._buffer.items():
      arr = np.array(values)
      mean = np.mean(arr)
      std = np.std(arr)
      fps = 1000 / mean if mean > 0 else 0
      log.info(f" {key:<20} mean={mean:6.1f} ms std={std:5.1f} fps={fps:5.1}")
    log.info(f"  samples: {len(next(iter(self._buffer.values())))}")
    self._buffer.clear()
    
  def close(self):
    if self.log_to_csv and self._rows:
      save_csv(self._rows, self.csv_path)