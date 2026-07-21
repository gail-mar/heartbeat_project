import pytest

from app import app


@pytest.fixture
def client():
    return app.test_client()


def test_get_index_shows_upload_form(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Analyze" in r.data
    assert b"CHROM" in r.data
    assert b"POS" in r.data
    assert b"pos_window_seconds" in r.data
    assert b"pos_overlap_percent" in r.data
    assert b"roi_mode" in r.data
    assert b"forehead_cheeks" in r.data


def test_post_without_file_shows_error(client):
    r = client.post("/", data={}, content_type="multipart/form-data")
    assert b"Please choose a video file" in r.data


def test_get_webcam_page_loads(client):
    r = client.get("/webcam")
    assert r.status_code == 200
    assert b"Live Webcam Heart Rate" in r.data


def test_webcam_start_resets_session(client):
    r = client.post("/api/webcam/start")
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


def test_webcam_frame_without_file_returns_400(client):
    r = client.post("/api/webcam/frame", data={}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_webcam_frame_rejects_unknown_method(client):
    import io
    fake_jpeg = io.BytesIO(b"not a real jpeg")
    r = client.post(
        "/api/webcam/frame",
        data={"frame": (fake_jpeg, "frame.jpg"), "method": "not_a_real_method"},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400


def test_webcam_frame_rejects_undecodable_image(client):
    import io
    fake_jpeg = io.BytesIO(b"not a real jpeg")
    r = client.post(
        "/api/webcam/frame",
        data={"frame": (fake_jpeg, "frame.jpg")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400


def test_webcam_frame_with_faceless_image_reports_no_face(client):
    import io
    import cv2
    import numpy as np

    frame = np.full((100, 100, 3), 150, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    r = client.post(
        "/api/webcam/frame",
        data={"frame": (io.BytesIO(buf.tobytes()), "frame.jpg")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["face_detected"] is False
    assert body["bpm"] is None


@pytest.mark.parametrize("method", ["green_baseline", "chrom", "pos"])
def test_post_faceless_video_shows_clean_error(client, faceless_video_path, method):
    with open(faceless_video_path, "rb") as f:
        r = client.post(
            "/",
            data={"video": (f, "test.mp4"), "method": method},
            content_type="multipart/form-data",
        )
    assert r.status_code == 200
    assert b"No face was detected" in r.data


def test_api_analyze_without_file_returns_400(client):
    r = client.post("/api/analyze", data={}, content_type="multipart/form-data")
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_api_analyze_unknown_method_returns_400(client, faceless_video_path):
    with open(faceless_video_path, "rb") as f:
        r = client.post(
            "/api/analyze",
            data={"video": (f, "test.mp4"), "method": "not_a_real_method"},
            content_type="multipart/form-data",
        )
    assert r.status_code == 400
    assert "error" in r.get_json()


@pytest.mark.parametrize("method", ["green_baseline", "chrom", "pos"])
def test_api_analyze_faceless_video_returns_422_with_json_error(client, faceless_video_path, method):
    with open(faceless_video_path, "rb") as f:
        r = client.post(
            "/api/analyze",
            data={"video": (f, "test.mp4"), "method": method},
            content_type="multipart/form-data",
        )
    assert r.status_code == 422
    body = r.get_json()
    assert "No face was detected" in body["error"]


def test_api_analyze_rejects_non_numeric_pos_window(client, faceless_video_path):
    with open(faceless_video_path, "rb") as f:
        r = client.post(
            "/api/analyze",
            data={"video": (f, "test.mp4"), "method": "pos", "pos_window_seconds": "not_a_number"},
            content_type="multipart/form-data",
        )
    assert r.status_code == 400
    assert "must be numbers" in r.get_json()["error"]


def test_api_analyze_accepts_custom_pos_window_and_overlap(client, faceless_video_path):
    # still a faceless video, so this only exercises that the custom values
    # are accepted and passed through without erroring before face-detection
    with open(faceless_video_path, "rb") as f:
        r = client.post(
            "/api/analyze",
            data={
                "video": (f, "test.mp4"),
                "method": "pos",
                "pos_window_seconds": "2.0",
                "pos_overlap_percent": "50",
            },
            content_type="multipart/form-data",
        )
    assert r.status_code == 422  # still fails on "no face", but AFTER accepting these params
    assert "No face was detected" in r.get_json()["error"]


def test_api_analyze_unknown_roi_mode_returns_400(client, faceless_video_path):
    with open(faceless_video_path, "rb") as f:
        r = client.post(
            "/api/analyze",
            data={"video": (f, "test.mp4"), "method": "chrom", "roi_mode": "not_a_real_roi"},
            content_type="multipart/form-data",
        )
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_api_analyze_accepts_forehead_cheeks_roi_mode(client, faceless_video_path):
    with open(faceless_video_path, "rb") as f:
        r = client.post(
            "/api/analyze",
            data={"video": (f, "test.mp4"), "method": "chrom", "roi_mode": "forehead_cheeks"},
            content_type="multipart/form-data",
        )
    assert r.status_code == 422  # still fails on "no face", but AFTER accepting roi_mode
    assert "No face was detected" in r.get_json()["error"]


def test_serve_video_404s_on_unknown_filename(client):
    r = client.get("/video/does-not-exist.mp4")
    assert r.status_code == 404


def test_api_analyze_crops_oversized_video_before_analysis(client, faceless_video_path, monkeypatch):
    # Force the "too big" branch with a tiny threshold instead of a real
    # multi-GB file, so this exercises the actual ffmpeg crop step.
    monkeypatch.setattr("app.DEFAULT_MAX_MB", 0.002)
    with open(faceless_video_path, "rb") as f:
        r = client.post(
            "/api/analyze",
            data={"video": (f, "test.mp4"), "method": "chrom"},
            content_type="multipart/form-data",
        )
    # still fails on "no face" (it's a faceless video), but AFTER cropping ran
    assert r.status_code == 422
    assert "No face was detected" in r.get_json()["error"]


