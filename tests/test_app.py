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


def test_post_without_file_shows_error(client):
    r = client.post("/", data={}, content_type="multipart/form-data")
    assert b"Please choose a video file" in r.data


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


