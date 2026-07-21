from flask import Flask, request, render_template, jsonify, Response, url_for, send_from_directory
import tempfile
import os

import cv2
import numpy as np

from heartbeat import (
    extract_rgb_signal,
    compute_bpm,
    compute_bpm_timeseries,
    METHODS,
    log_measurement,
    POS_WINDOW_SECONDS,
    render_svg_chart,
    ROI_MODES,
    render_roi_debug_frame,
    WebcamSession,
)
from video_tools.crop_to_size import crop_to_size, DEFAULT_MAX_MB

app = Flask(__name__)
# Raw uploads can be much bigger than what the analysis pipeline needs --
# anything over DEFAULT_MAX_MB gets losslessly cropped down to fit before
# analysis (see _run_analysis), so this is just a generous upper bound on
# what we'll even accept onto disk.
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024 * 1024  # 8 GB

# tempfile's default dir (usually /tmp) is often a small RAM-backed tmpfs --
# fine for tiny files, but multi-GB uploads can blow past it. /var/tmp is
# normal disk-backed storage, so uploads land there instead. Setting
# tempfile.tempdir (not just passing dir= to our own calls) also redirects
# Werkzeug's internal multipart upload buffering, which happens before our
# route code ever runs.
UPLOAD_TMP_DIR = "/var/tmp/heartbeat_uploads"
os.makedirs(UPLOAD_TMP_DIR, exist_ok=True)
tempfile.tempdir = UPLOAD_TMP_DIR

# Where analyzed videos are kept so they can be played back in the browser
# alongside a live BPM readout. This is a single-user dev tool, so we only
# ever keep the most recently analyzed video -- see _promote_to_playback.
PLAYBACK_DIR = os.path.join(UPLOAD_TMP_DIR, "playback")
os.makedirs(PLAYBACK_DIR, exist_ok=True)

# Rolling state for the live webcam BPM stream -- single-user dev tool, so
# one global session (reset each time "Start Camera" is clicked) is enough.
_webcam_session = WebcamSession()


def _promote_to_playback(analysis_path):
    """Move the analyzed video into PLAYBACK_DIR so /video/<filename> can
    serve it back, replacing whatever was kept from the previous analysis.
    Clears the directory rather than remembering the last filename in a
    variable, since that would reset (leaking old videos) every time the
    dev server's reloader restarts the process."""
    for old_name in os.listdir(PLAYBACK_DIR):
        old_path = os.path.join(PLAYBACK_DIR, old_name)
        if os.path.isfile(old_path):
            os.remove(old_path)
    filename = os.path.basename(analysis_path)
    os.rename(analysis_path, os.path.join(PLAYBACK_DIR, filename))
    return filename


def _run_analysis(video_file, method, reference_bpm, pos_window_seconds=None, pos_overlap_percent=None, roi_mode="full_face"):
    """Save an uploaded video, run the selected method, log the result.
    Returns (bpm, pulse_signal, fps, bpm_timeseries, video_filename,
    error_message) -- on error, everything but error_message is None.
    pos_window_seconds/pos_overlap_percent only apply when method == "pos"."""
    suffix = os.path.splitext(video_file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=UPLOAD_TMP_DIR) as tmp:
        tmp_path = tmp.name

    analysis_path = tmp_path
    was_cropped = False
    try:
        video_file.save(tmp_path)

        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        if size_mb > DEFAULT_MAX_MB:
            cropped_path = tmp_path + "_cropped" + suffix
            try:
                analysis_path = crop_to_size(tmp_path, cropped_path, DEFAULT_MAX_MB)
            except ValueError:
                raise
            except Exception as e:
                raise ValueError(f"Video is {size_mb:.0f} MB and could not be cropped to fit: {e}")
            was_cropped = analysis_path != tmp_path

        r_signal, g_signal, b_signal, fps, _regions_timeline, signal_timestamps = extract_rgb_signal(analysis_path, roi_mode=roi_mode)
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
        bpm_timeseries = compute_bpm_timeseries(pulse_signal, fps, timestamps=signal_timestamps)
    except (ValueError, OSError) as e:
        os.remove(tmp_path)
        if was_cropped and os.path.exists(analysis_path):
            os.remove(analysis_path)
        return None, None, None, None, None, str(e)

    # success: drop the raw upload if we analyzed a cropped copy instead of
    # it directly, then keep the analyzed video around for playback
    if was_cropped:
        os.remove(tmp_path)
    video_filename = _promote_to_playback(analysis_path)

    notes = ", ".join(filter(None, [
        f"roi_mode={roi_mode}" if roi_mode != "full_face" else "",
        "video_cropped_to_fit" if was_cropped else "",
    ]))
    log_measurement(video_file.filename, method, bpm, reference_bpm, notes=notes)
    return bpm, pulse_signal, fps, bpm_timeseries, video_filename, None


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

    roi_mode = request.form.get("roi_mode", "full_face")
    if roi_mode not in ROI_MODES:
        roi_mode = "full_face"

    try:
        pos_window_seconds = _parse_optional_float(request.form.get("pos_window_seconds"))
        pos_overlap_percent = _parse_optional_float(request.form.get("pos_overlap_percent"))
    except ValueError:
        return render_template("index.html", error="POS window/overlap must be numbers.")

    bpm, pulse_signal, fps, bpm_timeseries, video_filename, error = _run_analysis(
        video_file, method, reference_bpm, pos_window_seconds, pos_overlap_percent, roi_mode
    )
    if error:
        return render_template("index.html", error=error)
    return render_template(
        "index.html",
        bpm=bpm,
        chart_svg=render_svg_chart(pulse_signal, fps),
        bpm_timeseries=bpm_timeseries,
        video_url=url_for("serve_video", filename=video_filename),
    )


@app.route("/debug_roi", methods=["GET", "POST"])
def debug_roi():
    """Upload a video and see the detected face box (blue) plus the
    forehead (green) / cheek (red) regions sampled by roi_mode=forehead_cheeks,
    drawn on the first frame where a face is found."""
    if request.method == "GET":
        return render_template("debug_roi.html")

    video_file = request.files.get("video")
    if video_file is None or video_file.filename == "":
        return render_template("debug_roi.html", error="Please choose a video file.")

    suffix = os.path.splitext(video_file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=UPLOAD_TMP_DIR) as tmp:
        tmp_path = tmp.name

    try:
        video_file.save(tmp_path)
        png_bytes = render_roi_debug_frame(tmp_path)
    except (ValueError, OSError) as e:
        return render_template("debug_roi.html", error=str(e))
    finally:
        os.remove(tmp_path)

    if png_bytes is None:
        return render_template("debug_roi.html", error="No face was detected in this video.")

    return Response(png_bytes, mimetype="image/png")


@app.route("/video/<filename>")
def serve_video(filename):
    """Serve back the most recently analyzed video, for playback alongside
    the live BPM readout."""
    return send_from_directory(PLAYBACK_DIR, filename)


@app.route("/webcam")
def webcam_page():
    return render_template("webcam.html")


@app.route("/api/webcam/start", methods=["POST"])
def webcam_start():
    """Reset the rolling buffer -- called when the user clicks "Start
    Camera", so a previous session's samples don't bleed into a new one."""
    _webcam_session.reset()
    return jsonify({"ok": True})


@app.route("/api/webcam/frame", methods=["POST"])
def webcam_frame():
    """Accept one JPEG frame from the browser's webcam, sample its tracked
    face region, and return the current rolling BPM estimate (or None if
    there isn't enough data yet)."""
    image_file = request.files.get("frame")
    if image_file is None:
        return jsonify({"error": "Please provide a frame under the 'frame' form field."}), 400

    method = request.form.get("method", "green_baseline")
    if method not in METHODS:
        return jsonify({"error": f"Unknown method '{method}'. Valid options: {sorted(METHODS)}"}), 400

    roi_mode = request.form.get("roi_mode", "full_face")
    if roi_mode not in ROI_MODES:
        return jsonify({"error": f"Unknown roi_mode '{roi_mode}'. Valid options: {sorted(ROI_MODES)}"}), 400

    raw_bytes = np.frombuffer(image_file.read(), dtype=np.uint8)
    frame = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode frame."}), 400

    face_detected = _webcam_session.process_frame(frame, roi_mode)
    bpm = _webcam_session.current_bpm(method)
    return jsonify({
        "face_detected": face_detected,
        "bpm": bpm,
        "sample_count": len(_webcam_session.samples),
    })


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    video_file = request.files.get("video")
    if video_file is None or video_file.filename == "":
        return jsonify({"error": "Please provide a video file under the 'video' form field."}), 400

    method = request.form.get("method", "green_baseline")
    if method not in METHODS:
        return jsonify({"error": f"Unknown method '{method}'. Valid options: {sorted(METHODS)}"}), 400

    roi_mode = request.form.get("roi_mode", "full_face")
    if roi_mode not in ROI_MODES:
        return jsonify({"error": f"Unknown roi_mode '{roi_mode}'. Valid options: {sorted(ROI_MODES)}"}), 400

    reference_bpm = request.form.get("reference_bpm", "").strip()

    try:
        pos_window_seconds = _parse_optional_float(request.form.get("pos_window_seconds"))
        pos_overlap_percent = _parse_optional_float(request.form.get("pos_overlap_percent"))
    except ValueError:
        return jsonify({"error": "pos_window_seconds/pos_overlap_percent must be numbers."}), 400

    bpm, pulse_signal, fps, bpm_timeseries, video_filename, error = _run_analysis(
        video_file, method, reference_bpm, pos_window_seconds, pos_overlap_percent, roi_mode
    )
    if error:
        return jsonify({"error": error}), 422

    return jsonify({
        "bpm": bpm,
        "method": method,
        "roi_mode": roi_mode,
        "reference_bpm": float(reference_bpm) if reference_bpm else None,
        "bpm_timeseries": bpm_timeseries,
        "video_url": url_for("serve_video", filename=video_filename, _external=True),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5050)
