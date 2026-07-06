import numpy as np

from heartbeat import signal_to_svg_points, render_svg_chart


def test_signal_to_svg_points_returns_one_point_per_sample():
    signal = np.sin(np.linspace(0, 10, 50))
    points = signal_to_svg_points(signal)
    assert len(points.split()) == 50


def test_signal_to_svg_points_downsamples_long_signals():
    signal = np.sin(np.linspace(0, 10, 5000))
    points = signal_to_svg_points(signal, max_points=800)
    assert len(points.split()) == 800


def test_signal_to_svg_points_empty_signal_returns_empty_string():
    assert signal_to_svg_points([]) == ""


def test_signal_to_svg_points_flat_signal_does_not_crash():
    # a constant signal has zero variance -- must not divide by zero
    points = signal_to_svg_points([150.0] * 30)
    assert len(points.split()) == 30


def test_signal_to_svg_points_stays_within_bounds():
    signal = np.sin(np.linspace(0, 20, 200)) * 1000  # large amplitude
    width, height = 600, 150
    points = signal_to_svg_points(signal, width=width, height=height)
    for pair in points.split():
        x, y = map(float, pair.split(","))
        assert 0 <= x <= width
        assert 0 <= y <= height


def test_render_svg_chart_includes_axis_labels_and_polyline():
    fps = 30
    signal = np.sin(np.linspace(0, 20, 300))
    svg = render_svg_chart(signal, fps)
    assert "<polyline" in svg
    assert "Time (seconds)" in svg
    assert "Signal (normalized)" in svg
    # duration = 300 frames / 30 fps = 10s, so the last x-tick should show "10.0s"
    assert "10.0s" in svg


def test_render_svg_chart_empty_signal_returns_empty_string():
    assert render_svg_chart([], fps=30) == ""


def test_render_svg_chart_handles_missing_fps_gracefully():
    # fps=0 shouldn't crash the duration calculation (divide by zero)
    svg = render_svg_chart([150.0] * 30, fps=0)
    assert "<polyline" in svg
