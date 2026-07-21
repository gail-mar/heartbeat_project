import time

import numpy as np

from heartbeat import WebcamSession


def _blank_frame():
    return np.full((300, 300, 3), 150, dtype=np.uint8)


def test_process_frame_returns_false_on_faceless_frame():
    session = WebcamSession()
    assert session.process_frame(_blank_frame(), roi_mode="full_face") is False
    assert session.samples == []


def test_current_bpm_none_with_no_samples():
    session = WebcamSession()
    assert session.current_bpm("chrom") is None


def test_current_bpm_none_with_too_little_time_span():
    session = WebcamSession()
    now = time.time()
    # plenty of samples, but they only span 1 second -- under
    # MIN_SECONDS_FOR_BPM
    session.samples = [(now + i * 0.1, 150.0, 150.0, 150.0) for i in range(10)]
    assert session.current_bpm("chrom") is None


def test_reset_clears_samples_and_tracking_state():
    session = WebcamSession()
    session.samples = [(time.time(), 1.0, 2.0, 3.0)]
    session.last_face_box = (0, 0, 10, 10)
    session.reset()
    assert session.samples == []
    assert session.last_face_box is None
