import numpy as np
import pytest

from heartbeat import extract_rgb_signal, ROI_MODES
from heartbeat import haar_detection, mediapipe_landmarks
from heartbeat.detection import _mean_rgb_over_regions, _rect_to_quad, pick_tracked_index


@pytest.mark.parametrize("roi_mode", ROI_MODES)
def test_extract_rgb_signal_on_faceless_video_returns_empty(faceless_video_path, roi_mode):
    r, g, b, fps, regions_timeline, signal_timestamps = extract_rgb_signal(faceless_video_path, roi_mode=roi_mode)
    assert r == [] and g == [] and b == []
    assert regions_timeline == []
    assert signal_timestamps == []
    assert fps > 0


def test_extract_rgb_signal_raises_on_missing_file(tmp_path):
    with pytest.raises(ValueError, match="Could not open"):
        extract_rgb_signal(str(tmp_path / "does_not_exist.mp4"))


def test_extract_rgb_signal_raises_on_unknown_roi_mode(faceless_video_path):
    with pytest.raises(ValueError, match="Unknown roi_mode"):
        extract_rgb_signal(faceless_video_path, roi_mode="not_a_real_mode")


def test_detect_faces_returns_empty_on_faceless_frame():
    blank_gray = np.full((300, 300), 150, dtype=np.uint8)
    assert haar_detection.detect_faces(blank_gray) == []


def test_detect_landmarks_returns_empty_on_faceless_frame():
    blank_bgr = np.full((300, 300, 3), 150, dtype=np.uint8)
    assert mediapipe_landmarks.detect_landmarks(blank_bgr) == []


def test_pick_tracked_index_picks_largest_with_no_previous_target():
    # first-ever detection: no previous position to match against, so the
    # biggest face (most pixels) wins
    boxes = [(0, 0, 50, 50), (200, 200, 120, 120), (400, 0, 60, 60)]
    assert pick_tracked_index(boxes, previous_box=None) == 1


def test_pick_tracked_index_stays_on_the_same_person_not_the_new_larger_face():
    # we were tracking a face at (100,100,80,80); a second, much bigger
    # face shows up far away -- must keep tracking the original, not jump
    # to the bigger newcomer
    previous_box = (100, 100, 80, 80)
    boxes = [(500, 500, 300, 300), (105, 102, 78, 82)]
    assert pick_tracked_index(boxes, previous_box) == 1


def test_pick_tracked_index_returns_none_for_no_candidates():
    assert pick_tracked_index([], previous_box=(0, 0, 10, 10)) is None


def _synthetic_landmark_points(index_groups, n=478):
    """Build a fake list of (x, y) landmark points where each given set of
    indices is spread linearly across a known (x0, y0, x1, y1) box, so its
    bounding box is exactly that box. Lets us test forehead_cheek_regions'
    geometry without a real face image."""
    points = [(0, 0)] * n
    for idx_set, (x0, y0, x1, y1) in index_groups:
        idxs = sorted(idx_set)
        last = max(1, len(idxs) - 1)
        for i, idx in enumerate(idxs):
            t = i / last
            points[idx] = (int(x0 + t * (x1 - x0)), int(y0 + t * (y1 - y0)))
    return points


def _quad_bbox(quad):
    """(x, y, w, h) bounding box of a quad, for asserting on non-rotated
    (roll=0) test cases where the quad is an axis-aligned rectangle."""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return x0, y0, x1 - x0, y1 - y0


def test_forehead_cheek_regions_anchors_to_facial_feature_bands():
    # left/right eyes given separate, level (same y) positions so roll=0
    # and the returned quads are plain axis-aligned rectangles
    points = _synthetic_landmark_points([
        (mediapipe_landmarks._FACE_OVAL, (0, 0, 200, 300)),
        (mediapipe_landmarks._LEFT_EYEBROW | mediapipe_landmarks._RIGHT_EYEBROW, (60, 60, 140, 70)),
        (mediapipe_landmarks._LEFT_EYE, (60, 90, 100, 110)),
        (mediapipe_landmarks._RIGHT_EYE, (100, 90, 140, 110)),
        (mediapipe_landmarks._LIPS, (80, 220, 120, 230)),
        (mediapipe_landmarks._NOSE, (90, 120, 110, 200)),
    ])

    forehead, left_cheek, right_cheek = mediapipe_landmarks.forehead_cheek_regions(points)
    forehead, left_cheek, right_cheek = (_quad_bbox(q) for q in (forehead, left_cheek, right_cheek))

    for rx, ry, rw, rh in (forehead, left_cheek, right_cheek):
        assert rw > 0 and rh > 0
        assert 0 <= rx and rx + rw <= 200
        assert 0 <= ry and ry + rh <= 300

    # forehead sits above the eyebrows (y=60), not down at the hairline
    _, f_y, _, f_h = forehead
    assert f_y + f_h <= 60

    # cheeks sit between the eyes (bottom at y=110) and the lips (top at
    # y=220), not down near the mouth
    _, lc_y, _, lc_h = left_cheek
    _, rc_y, _, rc_h = right_cheek
    assert lc_y >= 110 and lc_y + lc_h <= 220
    assert rc_y >= 110 and rc_y + rc_h <= 220

    # cheeks are split by the nose (x 90-110): left cheek to its left,
    # right cheek to its right
    lc_x, _, lc_w, _ = left_cheek
    rc_x, _, _, _ = right_cheek
    assert lc_x + lc_w <= 90
    assert rc_x >= 110


def test_forehead_cheek_regions_compensates_for_head_tilt():
    # same layout as above, but the whole face rotated 20 degrees (a head
    # tilt) around the face center -- the regions should still land in the
    # same *relative* place on the face, not get thrown off by the tilt
    import math
    angle = math.radians(20)
    center = (100, 150)

    def rot(x, y):
        dx, dy = x - center[0], y - center[1]
        return (
            center[0] + dx * math.cos(angle) - dy * math.sin(angle),
            center[1] + dx * math.sin(angle) + dy * math.cos(angle),
        )

    base_points = _synthetic_landmark_points([
        (mediapipe_landmarks._FACE_OVAL, (0, 0, 200, 300)),
        (mediapipe_landmarks._LEFT_EYEBROW | mediapipe_landmarks._RIGHT_EYEBROW, (60, 60, 140, 70)),
        (mediapipe_landmarks._LEFT_EYE, (60, 90, 100, 110)),
        (mediapipe_landmarks._RIGHT_EYE, (100, 90, 140, 110)),
        (mediapipe_landmarks._LIPS, (80, 220, 120, 230)),
        (mediapipe_landmarks._NOSE, (90, 120, 110, 200)),
    ])
    tilted_points = [rot(x, y) for x, y in base_points]

    forehead, left_cheek, right_cheek = mediapipe_landmarks.forehead_cheek_regions(tilted_points)

    # the forehead quad's center should still sit roughly between the
    # (tilted) face-oval top and eyebrows, not somewhere thrown off by
    # the rotation (e.g. off to one side or overlapping the eyes)
    face_oval_pts = [tilted_points[i] for i in mediapipe_landmarks._FACE_OVAL]
    brow_pts = [tilted_points[i] for i in (mediapipe_landmarks._LEFT_EYEBROW | mediapipe_landmarks._RIGHT_EYEBROW)]
    face_cy = sum(p[1] for p in face_oval_pts) / len(face_oval_pts)
    brow_cy = sum(p[1] for p in brow_pts) / len(brow_pts)

    forehead_cx = sum(p[0] for p in forehead) / 4
    forehead_cy = sum(p[1] for p in forehead) / 4
    # roughly between the (rotated) face center and the eyebrows in the
    # rotated frame -- loose bounds since this is just a sanity check that
    # it didn't end up somewhere wildly wrong
    assert min(face_cy, brow_cy) - 50 <= forehead_cy <= max(face_cy, brow_cy) + 50
    assert 0 <= forehead_cx <= 250


def test_mean_rgb_over_regions_combines_multiple_patches():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[0:10, 0:10] = (10, 20, 30)   # BGR
    frame[50:60, 50:60] = (40, 50, 60)  # BGR

    regions = [_rect_to_quad(0, 0, 10, 10), _rect_to_quad(50, 50, 10, 10)]
    r, g, b = _mean_rgb_over_regions(frame, regions, frame_w=100, frame_h=100)

    # average of the two patches' R/G/B channels
    assert r == pytest.approx((30 + 60) / 2)
    assert g == pytest.approx((20 + 50) / 2)
    assert b == pytest.approx((10 + 40) / 2)


def test_mean_rgb_over_regions_returns_none_when_no_regions_given():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert _mean_rgb_over_regions(frame, [], frame_w=100, frame_h=100) is None
