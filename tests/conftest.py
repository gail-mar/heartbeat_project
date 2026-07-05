import cv2
import numpy as np
import pytest


@pytest.fixture
def faceless_video_path(tmp_path):
    """A short synthetic video with no face in it, for testing error paths."""
    path = str(tmp_path / "faceless.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30, (100, 100))
    for _ in range(90):
        writer.write(np.full((100, 100, 3), 150, dtype=np.uint8))
    writer.release()
    return path
