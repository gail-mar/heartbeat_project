from flask import Flask, request, render_template, jsonify
import tempfile
import os

from heartbeat import extract_rgb_signal, compute_bpm, METHODS, log_measurement

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2048 * 1024 * 1024  # 2 GB max upload (research datasets are often uncompressed and large)


def _run_analysis(video_file, method, reference_bpm):
    """Save an uploaded video, run the selected method, log the result.
    Returns (bpm, error_message) -- exactly one of which is None."""
    suffix = os.path.splitext(video_file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        video_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        r_signal, g_signal, b_signal, fps = extract_rgb_signal(tmp_path)
        if not g_signal:
            raise ValueError("No face was detected in this video.")
        pulse_signal = METHODS[method](r_signal, g_signal, b_signal, fps)
        bpm = compute_bpm(pulse_signal, fps)
    except ValueError as e:
        return None, str(e)
    finally:
        os.remove(tmp_path)

    log_measurement(video_file.filename, method, bpm, reference_bpm)
    return bpm, None


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

    bpm, error = _run_analysis(video_file, method, reference_bpm)
    if error:
        return render_template("index.html", error=error)
    return render_template("index.html", bpm=bpm)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    video_file = request.files.get("video")
    if video_file is None or video_file.filename == "":
        return jsonify({"error": "Please provide a video file under the 'video' form field."}), 400

    method = request.form.get("method", "green_baseline")
    if method not in METHODS:
        return jsonify({"error": f"Unknown method '{method}'. Valid options: {sorted(METHODS)}"}), 400

    reference_bpm = request.form.get("reference_bpm", "").strip()

    bpm, error = _run_analysis(video_file, method, reference_bpm)
    if error:
        return jsonify({"error": error}), 422

    return jsonify({
        "bpm": bpm,
        "method": method,
        "reference_bpm": float(reference_bpm) if reference_bpm else None,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5050)
