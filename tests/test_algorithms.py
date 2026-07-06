import numpy as np
import pytest

from heartbeat import compute_bpm, green_baseline_signal, chrom_signal, pos_signal


def _synthetic_signal(fps, duration_s, bpm, amplitude=1.0, offset=150):
    t = np.arange(fps * duration_s) / fps
    return offset + amplitude * np.sin(2 * np.pi * (bpm / 60) * t), t


def test_compute_bpm_recovers_known_frequency():
    fps = 30
    signal, _ = _synthetic_signal(fps, duration_s=10, bpm=72, amplitude=10)
    assert compute_bpm(list(signal), fps) == pytest.approx(72, abs=1.0)


def test_compute_bpm_raises_on_empty_signal():
    with pytest.raises(ValueError, match="No face"):
        compute_bpm([], fps=30)


def test_compute_bpm_raises_on_too_short_signal():
    with pytest.raises(ValueError, match="too short"):
        compute_bpm([150, 151, 150], fps=30)


def test_green_baseline_signal_is_just_the_green_channel():
    r, g, b = [1, 2, 3], [4, 5, 6], [7, 8, 9]
    result = green_baseline_signal(r, g, b, fps=30)
    assert list(result) == g


def _shared_motion_case(fps=30, duration_s=20, motion_bpm=54, true_bpm=78):
    """Build R/G/B traces where a big 'motion/lighting' wobble affects all
    three channels equally, and a smaller true pulse affects them differently
    (like real skin does) -- the setup used throughout this project to sanity
    check that an algorithm rejects shared noise instead of being fooled by it."""
    t = np.arange(fps * duration_s) / fps
    motion = 20 * np.sin(2 * np.pi * (motion_bpm / 60) * t)
    pulse = np.sin(2 * np.pi * (true_bpm / 60) * t)

    r = 150 + motion * 1.0 + pulse * 0.5
    g = 150 + motion * 1.0 + pulse * 1.0
    b = 150 + motion * 1.0 + pulse * 0.3
    return r, g, b, fps, motion_bpm, true_bpm


def test_green_baseline_is_fooled_by_shared_motion_artifact():
    r, g, b, fps, motion_bpm, true_bpm = _shared_motion_case()
    bpm = compute_bpm(green_baseline_signal(r, g, b, fps), fps)
    assert bpm == pytest.approx(motion_bpm, abs=1.0)


def test_chrom_rejects_shared_motion_artifact():
    r, g, b, fps, motion_bpm, true_bpm = _shared_motion_case()
    bpm = compute_bpm(chrom_signal(r, g, b, fps), fps)
    assert bpm == pytest.approx(true_bpm, abs=1.0)


def test_pos_rejects_shared_motion_artifact():
    r, g, b, fps, motion_bpm, true_bpm = _shared_motion_case()
    bpm = compute_bpm(pos_signal(r, g, b, fps), fps)
    assert bpm == pytest.approx(true_bpm, abs=1.0)


@pytest.mark.parametrize("window_seconds,step_frames", [
    (1.6, 1),    # default: maximum overlap
    (1.6, 24),   # ~50% overlap (half of a ~48-frame window at 30fps)
    (1.6, 48),   # ~0% overlap: non-overlapping back-to-back windows
    (3.0, 1),    # a longer window
])
def test_pos_signal_works_with_custom_window_and_overlap(window_seconds, step_frames):
    r, g, b, fps, motion_bpm, true_bpm = _shared_motion_case()
    bpm = compute_bpm(pos_signal(r, g, b, fps, window_seconds=window_seconds, step_frames=step_frames), fps)
    assert bpm == pytest.approx(true_bpm, abs=1.0)
