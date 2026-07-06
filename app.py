from flask import Flask, request, render_template, jsonify
import tempfile
import os

from heartbeat import (
    extract_rgb_signal,
    compute_bpm,
    METHODS,
    log_measurement,
    POS_WINDOW_SECONDS,
    render_svg_chart,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2048 * 1024 * 1024  # 2 GB max upload (research datasets are often uncompressed and large)


def _run_analysis(video_file, method, reference_bpm, pos_window_seconds=None, pos_overlap_percent=None):
    """Save an uploaded video, run the selected method, log the result.
    Returns (bpm, pulse_signal, fps, error_message) -- on error, the first
    three are None. pos_window_seconds/pos_overlap_percent only apply when
    method == "pos"."""
    suffix = os.path.splitext(video_file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        video_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        r_signal, g_signal, b_signal, fps = extract_rgb_signal(tmp_path)
        if not g_signal:
            raise ValueError("No face was detected in this video.")

        kwargs = {}
        if method == "pos":
            window_seconds = pos_window_seconds if pos_window_seconds is not None else POS_WINDOW_SECONDS
            kwargs["window_seconds"] = window_seconds
            if pos_overlap_percent is not None:
                window_len = max(2, round(window_seconds * fps))
                overlap_fraction = pos_overlap_percent / 100
                kwargs["step_frames"] = max(1, round(window_len * (1 - overlap_fraction)))

        pulse_signal = METHODS[method](r_signal, g_signal, b_signal, fps, **kwargs)
        bpm = compute_bpm(pulse_signal, fps)
    except ValueError as e:
        return None, None, None, str(e)
    finally:
        os.remove(tmp_path)

    log_measurement(video_file.filename, method, bpm, reference_bpm)
    return bpm, pulse_signal, fps, None


def _parse_optional_float(raw_value):
    raw_value = (raw_value or "").strip()
    return float(raw_value) if raw_value else None


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    video_file = request.files.get("video")
    if video_file is None or video_file.filename == "":
        return render_template("index.html", error="Please choose a video file.")

    reference_bpm = request.form.get("reference_bpm", "").strip()
    method = request.form.get("method", "green_baseline")
    if method not in METHODS:
        method = "green_baseline"

    try:
        pos_window_seconds = _parse_optional_float(request.form.get("pos_window_seconds"))
        pos_overlap_percent = _parse_optional_float(request.form.get("pos_overlap_percent"))
    except ValueError:
        return render_template("index.html", error="POS window/overlap must be numbers.")

    bpm, pulse_signal, fps, error = _run_analysis(video_file, method, reference_bpm, pos_window_seconds, pos_overlap_percent)
    if error:
        return render_template("index.html", error=error)
    return render_template("index.html", bpm=bpm, chart_svg=render_svg_chart(pulse_signal, fps))


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    video_file = request.files.get("video")
    if video_file is None or video_file.filename == "":
        return jsonify({"error": "Please provide a video file under the 'video' form field."}), 400

    method = request.form.get("method", "green_baseline")
    if method not in METHODS:
        return jsonify({"error": f"Unknown method '{method}'. Valid options: {sorted(METHODS)}"}), 400

    reference_bpm = request.form.get("reference_bpm", "").strip()

    try:
        pos_window_seconds = _parse_optional_float(request.form.get("pos_window_seconds"))
        pos_overlap_percent = _parse_optional_float(request.form.get("pos_overlap_percent"))
    except ValueError:
        return jsonify({"error": "pos_window_seconds/pos_overlap_percent must be numbers."}), 400

    bpm, pulse_signal, fps, error = _run_analysis(video_file, method, reference_bpm, pos_window_seconds, pos_overlap_percent)
    if error:
        return jsonify({"error": error}), 422

    return jsonify({
        "bpm": bpm,
        "method": method,
        "reference_bpm": float(reference_bpm) if reference_bpm else None,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5050)
