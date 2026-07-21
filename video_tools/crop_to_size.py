#!/usr/bin/env python3
"""Trim a video to fit under a target file size via a lossless stream copy
(no re-encoding), so the pixel colors rPPG reads stay untouched."""
import argparse
import json
import math
import os
import subprocess
import sys

DEFAULT_MAX_MB = 2000  # a bit under the app's 2048 MiB limit, for container overhead


def probe_duration_and_bitrate(path):
    """Return (duration_seconds, bitrate_bps) via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,bit_rate", "-of", "json", path,
        ],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(result.stdout)["format"]
    duration = float(info["duration"])
    bitrate = info.get("bit_rate")
    if bitrate is None:
        # some containers (e.g. certain .mov files) don't report an overall
        # bitrate -- fall back to file_size / duration
        bitrate = os.path.getsize(path) * 8 / duration
    else:
        bitrate = float(bitrate)
    return duration, bitrate


def probe_is_raw_video(path):
    """Return True if the video stream is an uncompressed/raw codec."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name", "-of", "json", path,
        ],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    return bool(streams) and streams[0].get("codec_name") == "rawvideo"


def crop_to_size(input_path, output_path, max_mb):
    duration, bitrate = probe_duration_and_bitrate(input_path)
    max_bytes = max_mb * 1024 * 1024
    max_duration = (max_bytes * 8) / bitrate

    if max_duration >= duration:
        print(f"Not cropped: '{input_path}' is {duration:.1f}s and already under {max_mb} MB. Output length: {duration:.1f}s (unchanged).")
        return input_path

    trimmed_duration = math.floor(max_duration)
    if trimmed_duration <= 0:
        raise ValueError(
            f"Bitrate too high ({bitrate / 1e6:.1f} Mbps) to fit any clip under {max_mb} MB "
            "with a lossless trim -- re-encoding would be needed instead."
        )

    if probe_is_raw_video(input_path):
        # Raw/uncompressed streams cause real problems on a plain stream
        # copy: classic AVI has a legacy ~2GB chunk-size limit that crashes
        # ffmpeg's muxer (heap corruption) on large raw streams, and NUT
        # silently mistags the pixel format on copy (bgr24 -> rgb555le),
        # corrupting every frame. Re-encoding with FFV1 (a lossless codec --
        # bit-exact, no compression artifacts) into Matroska sidesteps both.
        output_path = os.path.splitext(output_path)[0] + ".mkv"
        cmd = ["ffmpeg", "-y", "-i", input_path, "-t", str(trimmed_duration), "-c:v", "ffv1", "-level", "3", "-c:a", "copy", output_path]
    else:
        cmd = ["ffmpeg", "-y", "-i", input_path, "-t", str(trimmed_duration), "-c", "copy", output_path]

    subprocess.run(cmd, check=True)
    print(f"Cropped: '{input_path}' was {duration:.1f}s, trimmed to fit under {max_mb} MB. Output length: {trimmed_duration}s -> '{output_path}'.")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Trim a video to fit under a target file size.")
    parser.add_argument("input", help="Path to the source video.")
    parser.add_argument("-o", "--output", help="Path for the trimmed video (default: <input>_cropped<ext>).")
    parser.add_argument("--max-mb", type=float, default=DEFAULT_MAX_MB, help=f"Target max size in MB (default: {DEFAULT_MAX_MB}).")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"No such file: {args.input}")

    if args.output:
        output_path = args.output
    else:
        root, ext = os.path.splitext(args.input)
        output_path = f"{root}_cropped{ext}"

    crop_to_size(args.input, output_path, args.max_mb)


if __name__ == "__main__":
    main()
