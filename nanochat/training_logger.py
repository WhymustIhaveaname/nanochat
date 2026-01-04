"""
Training logger for saving step-by-step metrics to CSV.

Used for scaling law analysis and experiment tracking.

Usage:
    from nanochat.training_logger import TrainingLogger

    # Initialize (only on master process)
    logger = TrainingLogger("path/to/logs", run_name="d20_exp1")

    # Log each step - pass any dict, keys become columns
    logger.log({"step": 100, "train_loss": 2.345, "mfu": 45.2})

    # Close when done
    logger.close()
"""

import csv
import os
import time
from datetime import datetime


class TrainingLogger:
    """
    Logs training metrics to a CSV file for scaling law analysis.

    - Accepts any dict, keys become CSV columns (sorted alphabetically)
    - Automatically adds timestamp and walltime columns
    - First log() call determines the columns (subsequent calls must match)
    """

    def __init__(
        self,
        log_dir: str,
        run_name: str = "default",
        enabled: bool = True,
    ):
        """
        Initialize the training logger.

        Args:
            log_dir: Directory to save log files
            run_name: Run name for the filename
            enabled: If False, all operations are no-ops (useful for non-master processes)
        """
        self.enabled = enabled
        self._start_time = time.time()
        self._file = None
        self._writer = None
        self._columns = None
        self._closed = False

        if not enabled:
            self.log_path = None
            return

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Create unique filename with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{run_name}_{timestamp_str}.csv"
        self.log_path = os.path.join(log_dir, filename)

    def log(self, data: dict) -> None:
        """
        Log a row of metrics.

        Args:
            data: Dict of metrics. Keys become column names.
                  First call determines columns, subsequent calls must have same keys.
        """
        if not self.enabled or self._closed:
            return

        # Add automatic columns
        row = dict(data)  # copy to avoid modifying input
        row["_timestamp"] = datetime.now().isoformat()
        row["_walltime"] = time.time() - self._start_time

        # Format float values to 6 significant figures
        for key, value in row.items():
            if isinstance(value, float):
                row[key] = f"{value:.6g}"

        # First call: initialize file and write header
        if self._file is None:
            self._columns = sorted(row.keys())
            self._file = open(self.log_path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._columns)
            self._writer.writeheader()

        # Write row
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        """Close the log file."""
        if not self.enabled or self._closed or self._file is None:
            return
        self._file.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @property
    def path(self) -> str | None:
        """Return the path to the log file, or None if disabled."""
        return self.log_path
