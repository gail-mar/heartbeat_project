import numpy as np

CHART_WIDTH = 600
CHART_HEIGHT = 190
CHART_MAX_POINTS = 800  # downsample longer signals so the SVG stays light

_MARGIN_LEFT = 45
_MARGIN_RIGHT = 15
_MARGIN_TOP = 15
_MARGIN_BOTTOM = 35


def _normalize_and_downsample(signal, max_points):
    """Center the signal on zero, scale it to [-1, 1], and evenly downsample
    it if it's longer than max_points. Shared by both chart functions below."""
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return signal

    if len(signal) > max_points:
        indices = np.linspace(0, len(signal) - 1, max_points).astype(int)
        signal = signal[indices]

    signal = signal - np.mean(signal)
    peak = np.max(np.abs(signal))
    return signal / peak if peak > 1e-8 else signal


def signal_to_svg_points(signal, width=CHART_WIDTH, height=CHART_HEIGHT, max_points=CHART_MAX_POINTS):
    """Turn a pulse signal into a bare SVG <polyline points="..."> string,
    scaled to fill the full width x height, with no axes."""
    normalized = _normalize_and_downsample(signal, max_points)
    if normalized.size == 0:
        return ""

    n = len(normalized)
    margin = 10
    xs = np.linspace(0, width, n)
    ys = height / 2 - normalized * (height / 2 - margin)

    return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


def render_svg_chart(signal, fps, width=CHART_WIDTH, height=CHART_HEIGHT, max_points=CHART_MAX_POINTS):
    """Build the inner SVG markup for a pulse-signal chart WITH axes: an
    x-axis in seconds (using fps to convert frame count to time) and a
    y-axis showing the normalized signal amplitude. Returns a string of SVG
    elements meant to be embedded directly inside an <svg> tag."""
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return ""

    duration_s = len(signal) / fps if fps else 0
    normalized = _normalize_and_downsample(signal, max_points)

    plot_x0, plot_x1 = _MARGIN_LEFT, width - _MARGIN_RIGHT
    plot_y0, plot_y1 = _MARGIN_TOP, height - _MARGIN_BOTTOM
    plot_w, plot_h = plot_x1 - plot_x0, plot_y1 - plot_y0
    inner_margin = 5  # so the signal's own peaks don't touch the -1/1 gridlines

    n = len(normalized)
    xs = plot_x0 + np.linspace(0, plot_w, n)
    ys = plot_y0 + plot_h / 2 - normalized * (plot_h / 2 - inner_margin)
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    parts = [
        f'<line x1="{plot_x0}" y1="{plot_y0}" x2="{plot_x0}" y2="{plot_y1}" stroke="#555" stroke-width="1"/>',
        f'<line x1="{plot_x0}" y1="{plot_y1}" x2="{plot_x1}" y2="{plot_y1}" stroke="#555" stroke-width="1"/>',
    ]

    # y-axis ticks: -1, 0, 1 (the normalized signal range)
    for value, label in [(1, "1"), (0, "0"), (-1, "-1")]:
        y = plot_y0 + plot_h / 2 - value * (plot_h / 2 - inner_margin)
        parts.append(f'<line x1="{plot_x0 - 4}" y1="{y:.1f}" x2="{plot_x0}" y2="{y:.1f}" stroke="#555" stroke-width="1"/>')
        parts.append(f'<text x="{plot_x0 - 7}" y="{y:.1f}" fill="#aaa" font-size="9" text-anchor="end" dominant-baseline="middle">{label}</text>')

    # x-axis ticks: 5 evenly spaced timestamps across the video's duration
    tick_count = 5
    for i in range(tick_count):
        frac = i / (tick_count - 1)
        x = plot_x0 + frac * plot_w
        t = frac * duration_s
        parts.append(f'<line x1="{x:.1f}" y1="{plot_y1}" x2="{x:.1f}" y2="{plot_y1 + 4}" stroke="#555" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{plot_y1 + 15}" fill="#aaa" font-size="9" text-anchor="middle">{t:.1f}s</text>')

    parts.append(
        f'<text x="{(plot_x0 + plot_x1) / 2:.1f}" y="{height - 3}" fill="#ccc" font-size="10" text-anchor="middle">'
        f'Time (seconds)</text>'
    )
    mid_y = (plot_y0 + plot_y1) / 2
    parts.append(
        f'<text x="12" y="{mid_y:.1f}" fill="#ccc" font-size="10" text-anchor="middle" '
        f'transform="rotate(-90 12 {mid_y:.1f})">Signal (normalized)</text>'
    )

    parts.append(f'<polyline points="{points}" fill="none" stroke="#39ff14" stroke-width="1.5"/>')

    return "".join(parts)
