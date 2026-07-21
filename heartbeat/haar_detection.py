import cv2

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


def detect_faces(gray_frame):
    """Detect all faces in a grayscale frame, downscaled for speed. Returns
    a list of (x, y, w, h) boxes in the ORIGINAL frame's coordinates
    (possibly empty).

    This only gives rough bounding boxes -- no internal landmarks (eyes,
    nose, mouth), so it's used for roi_mode="full_face". For anything that
    needs to know where specific facial features are (forehead_cheeks), see
    mediapipe_landmarks.py instead. Callers that need to keep tracking the
    same person across frames (rather than whichever face is momentarily
    largest) should pick among these with detection.pick_tracked_index."""
    height, width = gray_frame.shape
    scale = min(1.0, DETECT_MAX_WIDTH / width)
    small = cv2.resize(gray_frame, (int(width * scale), int(height * scale))) if scale < 1.0 else gray_frame

    faces = FACE_CASCADE.detectMultiScale(small, scaleFactor=1.1, minNeighbors=5)
    return [(int(x / scale), int(y / scale), int(w / scale), int(h / scale)) for (x, y, w, h) in faces]
