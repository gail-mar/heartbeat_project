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

METHOD_NAME = "green_baseline"  # bump/rename when a new extraction method is added
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


def extract_heartbeat_signal(video_path):
    """Read a video and return (green_signal, fps): the average green
    brightness of the detected face in every frame, plus the video's fps."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    green_signal = []
    last_face = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            # largest detected face, in case of multiple detections
            face = max(faces, key=lambda f: f[2] * f[3])
            last_face = face
        else:
            face = last_face

        if face is None:
            # no face has ever been found yet; skip this frame
            continue

        x, y, w, h = face
        roi = frame[y:y + h, x:x + w]
        # green channel is index 1 in OpenCV's BGR ordering
        green_signal.append(float(np.mean(roi[:, :, 1])))

    cap.release()
    return green_signal, fps


def compute_bpm(green_signal, fps):
    """Turn the raw green-brightness trace into a BPM estimate using an FFT."""
    if not green_signal:
        raise ValueError("No face was detected in this video.")
    if len(green_signal) < fps * 2:
        raise ValueError("Video too short to estimate a heart rate.")

    signal = np.array(green_signal)
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

    suffix = os.path.splitext(video_file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        video_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        green_signal, fps = extract_heartbeat_signal(tmp_path)
        bpm = compute_bpm(green_signal, fps)
    except ValueError as e:
        return render_template("index.html", error=str(e))
    finally:
        os.remove(tmp_path)

    log_measurement(video_file.filename, METHOD_NAME, bpm, reference_bpm)
    return render_template("index.html", bpm=bpm)


if __name__ == "__main__":
    app.run(debug=True, port=5050)


