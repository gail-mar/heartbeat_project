import subprocess

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


@pytest.fixture
def raw_avi_video_path(tmp_path):
    """A tiny synthetic *uncompressed* AVI (rawvideo bgr24), solid red so
    tests can check the exact pixel value survives cropping. Large versions
    of this exact shape (AVI + raw video) used to crash ffmpeg's muxer with
    heap corruption when cropped, and remuxing to NUT (a since-reverted fix)
    silently mistagged the pixel format and corrupted every frame -- see
    video_tools/crop_to_size.py."""
    path = str(tmp_path / "raw.avi")
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:size=64x64:duration=2:rate=10",
            "-pix_fmt", "bgr24", "-c:v", "rawvideo", path,
        ],
        check=True, capture_output=True,
    )
    return path
