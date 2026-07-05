import cv2
import numpy as np

# Load OpenCV's pre-trained face detector
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Face detection is the expensive part of every frame, so we don't run it on
# every single frame, and we shrink the frame first (Haar cascades scan much
# faster on smaller images). The face barely moves between a handful of
# frames, so reusing the last known position in between is a fine tradeoff.
DETECT_EVERY_N_FRAMES = 5
DETECT_MAX_WIDTH = 400


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
