from flask import Flask, request, render_template
import tempfile
import os

from heartbeat import extract_rgb_signal, compute_bpm, METHODS, log_measurement

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2048 * 1024 * 1024  # 2 GB max upload (research datasets are often uncompressed and large)


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
        return render_template("index.html", error=str(e))
    finally:
        os.remove(tmp_path)

    log_measurement(video_file.filename, method, bpm, reference_bpm)
    return render_template("index.html", bpm=bpm)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
