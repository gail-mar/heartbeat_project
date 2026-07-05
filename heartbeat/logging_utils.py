import csv
import datetime
import os

# heartbeat/logging_utils.py -> parent of the heartbeat package -> project root
DEFAULT_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "measurements.csv")
LOG_FIELDS = ["timestamp", "video_filename", "method", "measured_bpm", "reference_bpm", "notes"]


def log_measurement(video_filename, method, measured_bpm, reference_bpm, notes="", log_path=DEFAULT_LOG_PATH):
    """Append one row to measurements.csv so different methods can be compared later."""
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "video_filename": video_filename,
            "method": method,
            "measured_bpm": measured_bpm,
            "reference_bpm": reference_bpm,
            "notes": notes,
        })
