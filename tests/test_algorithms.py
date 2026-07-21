import numpy as np
import pytest

from heartbeat import compute_bpm, compute_bpm_timeseries, green_baseline_signal, chrom_signal, pos_signal


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


def test_compute_bpm_timeseries_tracks_constant_bpm_throughout():
    fps = 30
    signal, _ = _synthetic_signal(fps, duration_s=20, bpm=72, amplitude=10)
    series = compute_bpm_timeseries(list(signal), fps, window_seconds=8, step_seconds=2)

    assert len(series) > 1
    timestamps = [t for t, _ in series]
    assert timestamps == sorted(timestamps)  # strictly increasing
    # an 8s/30fps window has ~7.5 BPM per FFT bin, so 72 (which doesn't land
    # exactly on a bin) quantizes to the nearest one -- allow for that
    for _, bpm in series:
        assert bpm == pytest.approx(72, abs=4.0)


def test_compute_bpm_timeseries_empty_on_signal_shorter_than_one_window():
    fps = 30
    signal, _ = _synthetic_signal(fps, duration_s=2, bpm=72, amplitude=10)
    series = compute_bpm_timeseries(list(signal), fps, window_seconds=8, step_seconds=2)
    assert series == []


def test_compute_bpm_timeseries_uses_real_timestamps_not_index_over_fps():
    # regression test: for variable-frame-rate video, sample index/fps
    # drifts from the real playback time -- if real per-sample timestamps
    # are given, they must be used instead of the index/fps assumption.
    fps = 30
    signal, _ = _synthetic_signal(fps, duration_s=20, bpm=72, amplitude=10)
    signal = list(signal)
    # "warped" timestamps: real time is double what index/fps would say
    # (as if the true frame rate were half of the nominal fps)
    warped_timestamps = [2 * i / fps for i in range(len(signal))]

    series = compute_bpm_timeseries(signal, fps, window_seconds=8, step_seconds=2, timestamps=warped_timestamps)

    assert len(series) > 1
    for t, _ in series:
        # with fps-based timing this would be well under 20s; with the
        # warped timestamps it should run out past the nominal duration
        assert t <= 2 * (len(signal) / fps)
    assert max(t for t, _ in series) > len(signal) / fps


def _signal_with_artifact_burst(fps=30, duration_s=24, true_bpm=72, artifact_bpm=150,
                                 burst_start_s=8, burst_end_s=16):
    """A constant true_bpm pulse throughout, plus a much stronger burst at
    an unrelated artifact_bpm confined to [burst_start_s, burst_end_s) --
    e.g. a motion/lighting glitch that briefly dominates the signal."""
    t = np.arange(fps * duration_s) / fps
    signal = 5 * np.sin(2 * np.pi * (true_bpm / 60) * t)
    burst_mask = (t >= burst_start_s) & (t < burst_end_s)
    signal = signal + np.where(burst_mask, 50 * np.sin(2 * np.pi * (artifact_bpm / 60) * t), 0.0)
    return signal


def test_compute_bpm_timeseries_ignores_transient_artifact_with_continuity_constraint():
    series = compute_bpm_timeseries(
        _signal_with_artifact_burst(), fps=30, window_seconds=8, step_seconds=2,
    )
    # every window should still track the true, constant heart rate --
    # none of them should jump to the artifact frequency (150 bpm)
    for _, bpm in series:
        assert bpm == pytest.approx(72, abs=8.0)


def test_compute_bpm_timeseries_would_be_fooled_without_the_constraint():
    # sanity check that the burst is actually strong enough to hijack the
    # naive (unconstrained) peak pick -- otherwise the test above wouldn't
    # be proving the continuity constraint does anything
    series = compute_bpm_timeseries(
        _signal_with_artifact_burst(), fps=30, window_seconds=8, step_seconds=2, max_jump=None,
    )
    assert any(bpm > 100 for _, bpm in series)


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
