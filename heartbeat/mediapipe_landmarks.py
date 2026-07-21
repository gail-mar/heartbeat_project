import math
import os

import cv2
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarksConnections,
)

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")

# Detect more than one face so a second person entering the frame can be
# told apart from whoever we're already tracking (see
# detection.pick_tracked_index), instead of mediapipe silently swapping
# which face it reports.
MAX_FACES = 5

_landmarker = FaceLandmarker.create_from_options(
    FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_faces=MAX_FACES,
    )
)


def _indices(connections):
    """Flatten a list of FaceLandmarksConnections.Connection into the set
    of landmark indices they touch."""
    idx = set()
    for c in connections:
        idx.add(c.start)
        idx.add(c.end)
    return idx


# Landmark groups anchored to actual facial features (eyebrows, eyes, lips,
# nose, face outline) rather than a guessed percentage of a bounding box.
_FACE_OVAL = _indices(FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL)
_LEFT_EYEBROW = _indices(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW)
_RIGHT_EYEBROW = _indices(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW)
_LEFT_EYE = _indices(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE)
_RIGHT_EYE = _indices(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE)
_LIPS = _indices(FaceLandmarksConnections.FACE_LANDMARKS_LIPS)
_NOSE = _indices(FaceLandmarksConnections.FACE_LANDMARKS_NOSE)


def detect_landmarks(frame_bgr):
    """Run MediaPipe FaceLandmarker on a BGR frame. Returns a list with one
    entry per detected face (up to MAX_FACES), each a list of (x, y) pixel
    coordinates for that face's landmarks. Empty list if no face was found."""
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _landmarker.detect(mp_image)
    return [
        [(int(p.x * w), int(p.y * h)) for p in face]
        for face in result.face_landmarks
    ]


def _bbox(points, idx_set):
    xs = [points[i][0] for i in idx_set]
    ys = [points[i][1] for i in idx_set]
    return min(xs), min(ys), max(xs), max(ys)


def _centroid(points, idx_set):
    xs = [points[i][0] for i in idx_set]
    ys = [points[i][1] for i in idx_set]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _rotate(point, pivot, angle):
    px, py = pivot
    x, y = point[0] - px, point[1] - py
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return (x * cos_a - y * sin_a + px, x * sin_a + y * cos_a + py)


def face_oval_bbox(points):
    """Bounding box (x, y, w, h) of the face oval landmarks, for drawing."""
    x0, y0, x1, y1 = _bbox(points, _FACE_OVAL)
    return x0, y0, x1 - x0, y1 - y0


def forehead_cheek_regions(points):
    """Derive forehead + cheek sample regions from face landmarks, anchored
    to the eyebrows/eyes/lips/face-oval instead of a guessed percentage of
    a bounding box, so they land on skin regardless of hairstyle. Head
    tilt/turn is compensated for by doing the whole layout in a
    face-aligned frame (rotated so the eye-line is horizontal) and then
    rotating the resulting rectangles back into image space, rather than
    assuming the face is upright and front-on. Returns
    [forehead, left_cheek, right_cheek], each a list of 4 (x, y) image-space
    corner points (a possibly-rotated rectangle, not an axis-aligned box)."""
    # FACE_LANDMARKS_LEFT_EYE/RIGHT_EYE are anatomical (the subject's own
    # left/right), which is mirrored relative to image x-coordinates for a
    # front-facing camera -- so anatomical-right can have a *larger* pixel
    # x than anatomical-left. Sort by actual image position instead of
    # trusting the anatomical label, so the eye-line vector always points
    # in a consistent image-space direction and the roll angle comes out
    # near 0 for a level, front-facing head instead of near +-180 degrees.
    eye_a, eye_b = _centroid(points, _LEFT_EYE), _centroid(points, _RIGHT_EYE)
    if eye_a[0] > eye_b[0]:
        eye_a, eye_b = eye_b, eye_a
    pivot = ((eye_a[0] + eye_b[0]) / 2, (eye_a[1] + eye_b[1]) / 2)
    roll = math.atan2(eye_b[1] - eye_a[1], eye_b[0] - eye_a[0])

    def aligned(idx_set):
        return [_rotate(points[i], pivot, -roll) for i in idx_set]

    face_pts, brow_pts, eye_pts = aligned(_FACE_OVAL), aligned(_LEFT_EYEBROW | _RIGHT_EYEBROW), aligned(_LEFT_EYE | _RIGHT_EYE)
    lips_pts, nose_pts = aligned(_LIPS), aligned(_NOSE)

    face_x0, face_x1 = min(p[0] for p in face_pts), max(p[0] for p in face_pts)
    face_y0, face_y1 = min(p[1] for p in face_pts), max(p[1] for p in face_pts)
    brow_y0 = min(p[1] for p in brow_pts)
    eye_y1 = max(p[1] for p in eye_pts)
    lips_y0 = min(p[1] for p in lips_pts)
    nose_x0, nose_x1 = min(p[0] for p in nose_pts), max(p[0] for p in nose_pts)

    face_w = face_x1 - face_x0

    # Forehead: the face-oval top to eyebrow gap is often tiny (mediapipe's
    # face-oval landmark hugs the brow, it doesn't reach the true hairline),
    # so a fixed fraction of *that* gap for clearance left the box touching
    # the eyebrows. Use most of the gap as clearance instead, and fill
    # whatever's left near the face-oval top with the sample band -- a thin
    # band with real clearance beats a taller one sitting on the eyebrows.
    forehead_top = face_y0
    forehead_bottom = max(forehead_top + 5, brow_y0 - 0.6 * (brow_y0 - face_y0))
    forehead_rect = (face_x0 + 0.25 * face_w, forehead_top, face_x0 + 0.75 * face_w, forehead_bottom)

    # Cheeks: below the eyes, above the lips, between the nose and the
    # face-oval edge -- the flat skin patch beside the nose. Margins scale
    # with face width (not the eye-to-lips gap, which can be similarly
    # small) so the boxes reliably clear the eyes and mouth. Each side uses
    # its own margin to the face edge, so a turned head (one side
    # foreshortened) narrows that side's box instead of distorting both.
    cheek_top = eye_y1 + 0.12 * face_w
    cheek_bottom = lips_y0 - 0.12 * face_w
    margin = 0.08 * face_w

    left_cheek_rect = (face_x0 + margin, cheek_top, nose_x0 - margin, cheek_bottom)
    right_cheek_rect = (nose_x1 + margin, cheek_top, face_x1 - margin, cheek_bottom)

    def to_image_quad(rect):
        x0, y0, x1, y1 = rect
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        return [tuple(int(round(c)) for c in _rotate(p, pivot, roll)) for p in corners]

    return [to_image_quad(r) for r in (forehead_rect, left_cheek_rect, right_cheek_rect)]
