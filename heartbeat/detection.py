import cv2
import numpy as np

from . import haar_detection, mediapipe_landmarks

# Detection (Haar for full_face, MediaPipe for forehead_cheeks) is the
# expensive part of every frame, so we don't run it on every single frame --
# the face barely moves between a handful of frames, so reusing the last
# known position/landmarks in between is a fine tradeoff.
DETECT_EVERY_N_FRAMES = 5

ROI_MODES = ("full_face", "forehead_cheeks")


def _box_center(box):
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def pick_tracked_index(candidate_boxes, previous_box):
    """Given face boxes detected in the current frame, pick which one is
    the person we're already tracking, so a second person entering the
    frame doesn't hijack tracking just by having a bigger or more
    confident detection. With no previous target yet (the very first
    detection in the video), picks the largest face instead -- "whoever's
    in front of the camera first, biggest wins." Returns an index into
    candidate_boxes, or None if candidate_boxes is empty."""
    if not candidate_boxes:
        return None
    if previous_box is None:
        return max(range(len(candidate_boxes)), key=lambda i: candidate_boxes[i][2] * candidate_boxes[i][3])
    px, py = _box_center(previous_box)
    return min(
        range(len(candidate_boxes)),
        key=lambda i: (_box_center(candidate_boxes[i])[0] - px) ** 2 + (_box_center(candidate_boxes[i])[1] - py) ** 2,
    )


def _rect_to_quad(x, y, w, h):
    # cv2.fillPoly rasterizes vertex coordinates inclusively, but w/h here
    # mean "w columns starting at x" (exclusive upper bound, like a Python
    # slice) -- so the far corner is (x+w-1, y+h-1), not (x+w, y+h), or
    # fillPoly would cover one extra row/column beyond the intended box.
    return [(x, y), (x + w - 1, y), (x + w - 1, y + h - 1), (x, y + h - 1)]


def _mean_rgb_over_regions(frame, regions, frame_w, frame_h):
    """Average B/G/R over all given regions combined. Each region is a
    polygon (list of (x, y) points -- e.g. the 4 corners of a possibly
    rotated rectangle), clipped to the frame's bounds. Returns (r, g, b)
    or None if nothing valid."""
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    for region in regions:
        if len(region) < 3:
            continue
        pts = np.array(
            [(min(max(x, 0), frame_w - 1), min(max(y, 0), frame_h - 1)) for x, y in region],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pts], 255)

    pixels = frame[mask == 255]
    if pixels.size == 0:
        return None

    b, g, r = pixels[:, 0].mean(), pixels[:, 1].mean(), pixels[:, 2].mean()
    return float(r), float(g), float(b)


def render_roi_debug_frame(video_path):
    """Find the first frame with a detected face, draw the face oval (blue)
    and the forehead_cheeks sample regions (forehead green, cheeks red) on
    it, and return PNG-encoded bytes. Returns None if no face is found."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    points = None
    frame = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                frame = None
                break
            faces = mediapipe_landmarks.detect_landmarks(frame)
            if faces:
                # just a single-frame preview, so no tracking history to
                # match against -- show whichever face is largest
                bboxes = [mediapipe_landmarks.face_oval_bbox(pts) for pts in faces]
                largest = max(range(len(faces)), key=lambda i: bboxes[i][2] * bboxes[i][3])
                points = faces[largest]
                break
    finally:
        cap.release()

    if points is None or frame is None:
        return None

    annotated = frame.copy()

    fx, fy, fw, fh = mediapipe_landmarks.face_oval_bbox(points)
    cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)

    forehead, left_cheek, right_cheek = mediapipe_landmarks.forehead_cheek_regions(points)
    for quad, color in (
        (forehead, (0, 255, 0)),
        (left_cheek, (0, 0, 255)),
        (right_cheek, (0, 0, 255)),
    ):
        cv2.polylines(annotated, [np.array(quad, dtype=np.int32)], isClosed=True, color=color, thickness=2)

    ok, buffer = cv2.imencode(".png", annotated)
    return buffer.tobytes() if ok else None


def extract_rgb_signal(video_path, roi_mode="full_face"):
    """Read a video and return (r_signal, g_signal, b_signal, fps,
    regions_timeline, signal_timestamps): the average red/green/blue
    brightness of the detected face (or a sub-region of it, per roi_mode)
    in every frame, plus a sparse (timestamp_seconds, [region, ...])
    history of the sampled region(s) each time detection re-ran (for
    drawing an overlay synced to video playback), plus each r/g/b sample's
    own real timestamp (signal_timestamps[i] is when r_signal[i] etc. was
    captured -- not i/fps, since that assumes evenly-spaced frames, which
    doesn't hold for variable-frame-rate video). Each region is a polygon:
    a list of 4 (x, y) corner points (a possibly-rotated rectangle for
    forehead_cheeks, an axis-aligned one for full_face)."""
    if roi_mode not in ROI_MODES:
        raise ValueError(f"Unknown roi_mode '{roi_mode}'. Valid options: {ROI_MODES}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    r_signal, g_signal, b_signal = [], [], []
    signal_timestamps = []
    regions_timeline = []
    last_regions = None
    last_face_box = None  # the tracked person's box, for matching across frames -- see pick_tracked_index
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % DETECT_EVERY_N_FRAMES == 0:
            frame_h, frame_w = frame.shape[:2]
            updated_regions = None

            if roi_mode == "forehead_cheeks":
                faces = mediapipe_landmarks.detect_landmarks(frame)
                candidate_boxes = [mediapipe_landmarks.face_oval_bbox(pts) for pts in faces]
                picked = pick_tracked_index(candidate_boxes, last_face_box)
                if picked is not None:
                    last_face_box = candidate_boxes[picked]
                    updated_regions = mediapipe_landmarks.forehead_cheek_regions(faces[picked])
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                candidate_boxes = haar_detection.detect_faces(gray)
                picked = pick_tracked_index(candidate_boxes, last_face_box)
                if picked is not None:
                    x, y, w, h = candidate_boxes[picked]
                    last_face_box = (x, y, w, h)
                    x, y = max(0, x), max(0, y)
                    w, h = min(w, frame_w - x), min(h, frame_h - y)
                    if w > 0 and h > 0:
                        updated_regions = [_rect_to_quad(x, y, w, h)]

            if updated_regions is not None:
                last_regions = updated_regions
                # Use the frame's actual decoded timestamp, not frame_idx/fps
                # -- for variable-frame-rate video (common from webcam
                # capture tools), frames aren't evenly spaced in time, so a
                # constant-rate assumption drifts from the real timestamp a
                # browser's <video currentTime> would report for this same
                # frame, throwing off which region gets shown during playback.
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                regions_timeline.append((round(timestamp, 2), last_regions))

        frame_idx += 1

        if last_regions is None:
            # no face has ever been found yet; skip this frame
            continue

        frame_h, frame_w = frame.shape[:2]
        result = _mean_rgb_over_regions(frame, last_regions, frame_w, frame_h)
        if result is None:
            continue

        r_val, g_val, b_val = result
        r_signal.append(r_val)
        g_signal.append(g_val)
        b_signal.append(b_val)
        signal_timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

    cap.release()
    return r_signal, g_signal, b_signal, fps, regions_timeline, signal_timestamps
