import time

import cv2

from . import haar_detection, mediapipe_landmarks
from .algorithms import METHODS, compute_bpm
from .detection import _mean_rgb_over_regions, _rect_to_quad, pick_tracked_index

# How much recent history to keep for a live BPM estimate. Old samples
# past this are dropped, so the readout tracks the subject's *current*
# heart rate rather than averaging over the whole session.
BUFFER_SECONDS = 30

# Need enough seconds of real data before an FFT-based BPM estimate is
# remotely reliable, regardless of how many samples that spans.
MIN_SECONDS_FOR_BPM = 4


class WebcamSession:
    """Rolling state for one live webcam BPM stream: a single-user dev
    tool, so there's just one active session at a time (see app.py)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.samples = []  # [(timestamp, r, g, b), ...], oldest first
        self.last_face_box = None

    def process_frame(self, frame_bgr, roi_mode):
        """Detect the tracked face in this frame and append its average
        color to the rolling buffer. Returns True if a face was found."""
        now = time.time()
        frame_h, frame_w = frame_bgr.shape[:2]

        if roi_mode == "forehead_cheeks":
            faces = mediapipe_landmarks.detect_landmarks(frame_bgr)
            candidate_boxes = [mediapipe_landmarks.face_oval_bbox(pts) for pts in faces]
            picked = pick_tracked_index(candidate_boxes, self.last_face_box)
            if picked is None:
                return False
            self.last_face_box = candidate_boxes[picked]
            regions = mediapipe_landmarks.forehead_cheek_regions(faces[picked])
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            candidate_boxes = haar_detection.detect_faces(gray)
            picked = pick_tracked_index(candidate_boxes, self.last_face_box)
            if picked is None:
                return False
            x, y, w, h = candidate_boxes[picked]
            self.last_face_box = (x, y, w, h)
            x, y = max(0, x), max(0, y)
            w, h = min(w, frame_w - x), min(h, frame_h - y)
            if w <= 0 or h <= 0:
                return False
            regions = [_rect_to_quad(x, y, w, h)]

        result = _mean_rgb_over_regions(frame_bgr, regions, frame_w, frame_h)
        if result is None:
            return False

        r, g, b = result
        self.samples.append((now, r, g, b))
        cutoff = now - BUFFER_SECONDS
        self.samples = [s for s in self.samples if s[0] >= cutoff]
        return True

    def current_bpm(self, method):
        """Estimate BPM from the rolling buffer, or None if there isn't
        enough data yet. The sampling rate is measured from the actual
        arrival times of frames (subject to real network/processing
        jitter), not assumed."""
        if len(self.samples) < 2:
            return None

        timestamps = [s[0] for s in self.samples]
        duration = timestamps[-1] - timestamps[0]
        if duration < MIN_SECONDS_FOR_BPM:
            return None

        avg_fps = len(self.samples) / duration
        r = [s[1] for s in self.samples]
        g = [s[2] for s in self.samples]
        b = [s[3] for s in self.samples]

        pulse_signal = METHODS[method](r, g, b, avg_fps)
        try:
            return compute_bpm(pulse_signal, avg_fps)
        except ValueError:
            return None
