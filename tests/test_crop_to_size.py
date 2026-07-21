import os

import cv2

from video_tools.crop_to_size import crop_to_size


def test_crop_to_size_leaves_small_video_unchanged(faceless_video_path, tmp_path):
    output_path = str(tmp_path / "out.mp4")
    result = crop_to_size(faceless_video_path, output_path, max_mb=2000)
    assert result == faceless_video_path


def test_crop_to_size_reencodes_raw_avi_losslessly_instead_of_crashing(raw_avi_video_path, tmp_path):
    # Regression test: cropping a raw/uncompressed AVI used to crash
    # ffmpeg's muxer with heap corruption when writing a large AVI output.
    # A later fix (remux to .nut) avoided the crash but silently mistagged
    # the pixel format, turning every frame black/garbage. It should now
    # re-encode with lossless FFV1 into .mkv and preserve the exact pixels.
    output_path = str(tmp_path / "out.avi")
    result = crop_to_size(raw_avi_video_path, output_path, max_mb=0.15)
    assert result.endswith(".mkv")
    assert os.path.isfile(result)

    cap = cv2.VideoCapture(result)
    ret, frame = cap.read()
    cap.release()
    assert ret
    # source was solid red (bgr24: B=0, G=0, R=255) -- must survive exactly
    b, g, r = frame[0, 0]
    assert (int(b), int(g)) == (0, 0)
    assert int(r) > 250
