
# Heartbeat Project

Estimate a person's heart rate (BPM) from an ordinary video of their face — no special hardware, just a webcam or phone video. This is a technique called **remote photoplethysmography (rPPG)**.

## How it works, in plain terms

Every time your heart beats, it pushes a little extra blood into the tiny blood vessels under your skin. That extra blood makes your skin absorb a tiny bit more light for a split second, then a bit less as the pulse passes. A camera recording enough frames per second can pick up this flicker as a very faint, repeating change in color on your face — far too subtle to see with your eyes, but visible in the numbers.

The hard part isn't finding this flicker — it's that everyday things (moving your head slightly, lighting changing, auto-exposure adjusting) cause color changes on your face that are much bigger than the real heartbeat signal. Most of the engineering here is about telling the real signal apart from that noise.

## The three methods

This project implements three increasingly sophisticated ways of pulling a heart rate out of a video, selectable from a dropdown in the app:

| Method | Idea | Robustness |
|---|---|---|
| **Baseline (green channel)** | Track the brightness of just the green channel on the face and look for a repeating wobble. Simplest possible approach. | Fragile — easily fooled by motion or lighting changes, since it has no way to tell them apart from a real pulse. |
| **CHROM** (de Haan & Jeanne, 2013) | Combine red, green, and blue together in a way that cancels out changes affecting all three colors similarly (motion, lighting), leaving mostly the pulse behind. | Much more resistant to noise than the baseline. |
| **POS** (Wang et al., 2016) | A refinement of the same idea as CHROM, using a different color-combination formula, computed over short overlapping ~1.6s time windows (not one average for the whole video) so it keeps adapting to local conditions. | Generally the most robust of the three in research literature, though our real-world tests below show it isn't automatically better in every case. |

All three end with the same step: an FFT (a standard tool that finds repeating rhythms in a signal) picks out the strongest frequency within a plausible human heart-rate range (42–240 BPM) and reports it as BPM.

## Real-world test results so far

Every analysis run gets logged to [`measurements.csv`](measurements.csv), including an optional "actual BPM" (from a smartwatch or manual pulse count) for comparison. Results collected so far:

| Video | Method | Measured BPM | Reference BPM | Notes |
|---|---|---|---|---|
| (early test) | baseline | 55 | 82 | Way off |
| (early test) | baseline | 42.9 | 64 | Landed right at the algorithm's search-range floor — a sign it didn't find a real heartbeat at all |
| WIN...Pro.mp4 | baseline | 42.9 | 64 | Same video, confirms baseline is unreliable |
| WIN...Pro.mp4 | chrom | 62.2 | 64 | Close — big improvement over baseline |
| WIN...Pro.mp4 | pos (before windowing fix) | 81.5 | 64 | Worse than CHROM — motivated switching POS to the proper sliding-window version |
| WIN...Pro.mp4 | pos (after windowing fix) | 62.2 | 64 | Matches CHROM once implemented correctly |
| Video Project.mp4 | pos | 106.3 | 75 | Way off — a reminder that no method here is reliable on every video yet |

**Takeaway so far:** CHROM and POS are clear improvements over the naive baseline, but neither is bulletproof — accuracy still depends a lot on the specific video (lighting, stillness, video quality). More test videos and comparisons are needed before trusting any single method's output.

## Project structure

```
heartbeat_project/
├── heartbeat/               # the core rPPG logic, as an installable Python package
│   ├── __init__.py          # public API: extract_rgb_signal, compute_bpm, METHODS, log_measurement...
│   ├── detection.py         # face detection + per-frame R/G/B signal extraction
│   ├── algorithms.py        # baseline / CHROM / POS + the FFT-based BPM calculation
│   └── logging_utils.py     # appends every run's result to measurements.csv
├── app.py                   # Flask app: web upload form + JSON API, both built on the heartbeat package
├── templates/index.html     # the web upload form
├── tests/                   # pytest test suite (see below)
├── pyproject.toml           # package metadata (`pip install -e .`)
├── measurements.csv         # log of every analysis run, for comparing methods over time
└── requirements / venv      # see Setup below
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -e .[web,test]      # installs the heartbeat package + Flask + pytest
```

## Running the app

```bash
source venv/bin/activate
python app.py
```

Then open **http://localhost:5050** in your browser.

- Upload a short video of a face (steady lighting and minimal movement give the best results).
- Pick an algorithm: Baseline, CHROM, or POS.
- Optionally enter the "actual BPM" from another device (smartwatch, pulse oximeter, manual count) — this gets logged alongside the result in `measurements.csv` for comparison.

## Using the JSON API

Besides the web form, there's a JSON endpoint for programmatic use:

```bash
curl -X POST http://localhost:5050/api/analyze \
  -F "video=@myvideo.mp4" \
  -F "method=chrom" \
  -F "reference_bpm=70"
```

**Request (multipart form-data):**
- `video` (required) — the video file
- `method` (optional) — `green_baseline` (default), `chrom`, or `pos`
- `reference_bpm` (optional) — logged alongside the result for later comparison

**Response on success (200):**
```json
{"bpm": 62.2, "method": "chrom", "reference_bpm": 70.0}
```

**Response on error** (400 for a bad request, e.g. missing file or unknown method; 422 if the video couldn't be processed, e.g. no face detected):
```json
{"error": "No face was detected in this video."}
```

## Running the tests

```bash
source venv/bin/activate
python -m pytest -v
```

The test suite covers:
- The BPM/FFT math, including a synthetic "shared motion artifact" scenario that verifies CHROM and POS actually reject noise that fools the baseline (this mirrors what we saw happen in real videos above)
- Face-detection edge cases (e.g. a video with no face in it)
- Both the web form and the JSON API's error handling

## Known limitations

- **Accuracy is inconsistent** — see the results table above. CHROM/POS help, but aren't reliable on every video yet.
- **Uncompressed/raw video formats can crash OpenCV.** Some research datasets (e.g. UBFC-rPPG) store video as raw uncompressed AVI, which has caused a hard crash (segfault) in this OpenCV build when reading frames — this is unresolved. Regular compressed video (phone/webcam recordings, .mp4) works fine.
- **Large uploads** are capped at 2GB (`MAX_CONTENT_LENGTH` in `app.py`).
- **Face detection** uses OpenCV's Haar cascade, which is fast but less accurate than modern deep-learning face detectors, especially at odd angles or poor lighting.
- Processing time scales with video length; face detection runs every 5th frame on a downscaled image to keep this reasonable, but very long videos will still take a while.
