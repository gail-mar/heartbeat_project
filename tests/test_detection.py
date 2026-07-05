from heartbeat import extract_rgb_signal


def test_extract_rgb_signal_on_faceless_video_returns_empty(faceless_video_path):
    r, g, b, fps = extract_rgb_signal(faceless_video_path)
    assert r == [] and g == [] and b == []
    assert fps > 0


def test_extract_rgb_signal_raises_on_missing_file(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="Could not open"):
        extract_rgb_signal(str(tmp_path / "does_not_exist.mp4"))
