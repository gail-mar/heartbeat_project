from flask import Flask, request, render_template
import cv2
import numpy as np
import tempfile
import os
import csv
import datetime

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB max upload

# Load OpenCV's pre-trained face detector
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

MIN_BPM = 42
MAX_BPM = 240

# Face detection is the expensive part of every frame, so we don't run it on
# every single frame, and we shrink the frame first (Haar cascades scan much
# faster on smaller images). The face barely moves between a handful of
# frames, so reusing the last known position in between is a fine tradeoff.
DETECT_EVERY_N_FRAMES = 5
DETECT_MAX_WIDTH = 400

MEASUREMENTS_LOG_PATH = os.path.join(os.path.dirname(__file__), "measurements.csv")
LOG_FIELDS = ["timestamp", "video_filename", "method", "measured_bpm", "reference_bpm", "notes"]


def log_measurement(video_filename, method, measured_bpm, reference_bpm, notes=""):
    """Append one row to measurements.csv so different methods can be compared later."""
    file_exists = os.path.isfile(MEASUREMENTS_LOG_PATH)
    with open(MEASUREMENTS_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "video_filename": video_filename,
            "method": method,
            "measured_bpm": measured_bpm,
            "reference_bpm": reference_bpm,
            "notes": notes,
        })


def _detect_face(gray_frame):
    """Detect the largest face in a grayscale frame, downscaled for speed.
    Returns (x, y, w, h) in the ORIGINAL frame's coordinates, or None."""
    height, width = gray_frame.shape
    scale = min(1.0, DETECT_MAX_WIDTH / width)
    small = cv2.resize(gray_frame, (int(width * scale), int(height * scale))) if scale < 1.0 else gray_frame

    faces = FACE_CASCADE.detectMultiScale(small, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return (int(x / scale), int(y / scale), int(w / scale), int(h / scale))


def extract_rgb_signal(video_path):
    """Read a video and return (r_signal, g_signal, b_signal, fps): the
    average red/green/blue brightness of the detected face in every frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    r_signal, g_signal, b_signal = [], [], []
    last_face = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % DETECT_EVERY_N_FRAMES == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = _detect_face(gray)
            if detected is not None:
                last_face = detected

        frame_idx += 1

        if last_face is None:
            # no face has ever been found yet; skip this frame
            continue

        frame_h, frame_w = frame.shape[:2]
        x, y, w, h = last_face
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame_w - x), min(h, frame_h - y)
        if w <= 0 or h <= 0:
            continue

        roi = frame[y:y + h, x:x + w]
        # OpenCV uses BGR ordering
        b_signal.append(float(np.mean(roi[:, :, 0])))
        g_signal.append(float(np.mean(roi[:, :, 1])))
        r_signal.append(float(np.mean(roi[:, :, 2])))

    cap.release()
    return r_signal, g_signal, b_signal, fps


def green_baseline_signal(r_signal, g_signal, b_signal, fps):
    """Simplest possible pulse signal: just the raw green channel."""
    return np.array(g_signal)


def chrom_signal(r_signal, g_signal, b_signal, fps):
    """CHROM (de Haan & Jeanne, 2013): combine R/G/B into two chrominance
    signals that lighting and motion affect similarly, then subtract them
    out to leave mostly the pulse behind."""
    r, g, b = np.array(r_signal), np.array(g_signal), np.array(b_signal)

    # normalize each channel by its own average brightness first, so the
    # combination below isn't dominated by one channel just being brighter
    rn, gn, bn = r / np.mean(r), g / np.mean(g), b / np.mean(b)

    x = 3 * rn - 2 * gn
    y = 1.5 * rn + gn - 1.5 * bn

    std_y = np.std(y)
    alpha = np.std(x) / std_y if std_y > 1e-8 else 0
    return x - alpha * y


def pos_signal(r_signal, g_signal, b_signal, fps):
    """POS (Wang et al., 2016), sliding-window version matching the original
    paper: instead of normalizing against one average over the whole video,
    slide a short window (1.6s) across the signal, normalize/project each
    window against its OWN local statistics, and overlap-add the results.
    This keeps the signal responsive to lighting/motion that drifts over
    the course of the video, which a single global average would miss."""
    rgb = np.stack([np.array(r_signal), np.array(g_signal), np.array(b_signal)], axis=1)  # frames x 3
    n_frames = rgb.shape[0]
    window_len = max(2, int(round(1.6 * fps)))

    h = np.zeros(n_frames)
    for n in range(n_frames):
        m = n - window_len
        if m < 0:
            continue

        window = rgb[m:n, :]  # window_len x 3
        normalized = window / np.mean(window, axis=0)  # each channel vs its OWN mean in this window

        s1 = normalized[:, 1] - normalized[:, 2]                              # Gn - Bn
        s2 = -2 * normalized[:, 0] + normalized[:, 1] + normalized[:, 2]      # -2Rn + Gn + Bn

        std_s2 = np.std(s2)
        alpha = np.std(s1) / std_s2 if std_s2 > 1e-8 else 0
        window_signal = s1 + alpha * s2

        # overlap-add: each frame gets contributions from every window that covered it
        h[m:n] += window_signal - np.mean(window_signal)

    return h


METHODS = {
    "green_baseline": green_baseline_signal,
    "chrom": chrom_signal,
    "pos": pos_signal,
}


def compute_bpm(signal, fps):
    """Turn a pulse-like signal into a BPM estimate using an FFT."""
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        raise ValueError("No face was detected in this video.")
    if len(signal) < fps * 2:
        raise ValueError("Video too short to estimate a heart rate.")

    signal = signal - np.mean(signal)  # remove the constant brightness offset
    signal = signal * np.hamming(len(signal))  # taper edges to reduce FFT artifacts

    fft_magnitudes = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fps)
    bpm_per_freq = freqs * 60

    valid = (bpm_per_freq >= MIN_BPM) & (bpm_per_freq <= MAX_BPM)
    if not np.any(valid):
        raise ValueError("No plausible heart rate found in this video.")

    peak_index = np.argmax(fft_magnitudes[valid])
    bpm = bpm_per_freq[valid][peak_index]
    return round(float(bpm), 1)


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


