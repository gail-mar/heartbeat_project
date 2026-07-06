import numpy as np

MIN_BPM = 42
MAX_BPM = 240


def green_baseline_signal(r_signal, g_signal, b_signal, fps):
    """Simplest possible pulse signal: just the raw green channel."""
    return np.array(g_signal)


def chrom_signal(r_signal, g_signal, b_signal, fps):
    """CHROM (de Haan & Jeanne, 2013): combine R/G/B into two chrominance
    signals that lighting and motion affect similarly, then subtract them
    out to leave mostly the pulse behind."""
    r, g, b = np.array(r_signal), np.array(g_signal), np.array(b_signal)

    # normalize each channel by its own average brightness first, so the
    # combination below isn't dominated by one channel just being brighter
    rn, gn, bn = r / np.mean(r), g / np.mean(g), b / np.mean(b)

    x = 3 * rn - 2 * gn
    y = 1.5 * rn + gn - 1.5 * bn

    std_y = np.std(y)
    alpha = np.std(x) / std_y if std_y > 1e-8 else 0
    return x - alpha * y


# How long each sliding window is, in seconds. Longer = more stable
# normalization statistics per window, but slower to react to changes.
POS_WINDOW_SECONDS = 1.6

# How many frames the window advances each step. 1 = maximum overlap (every
# frame gets a contribution from `window_len` different windows, the most
# responsive but most compute). Set this to `window_len` for non-overlapping
# chunks (0% overlap), or `window_len // 2` for the classic "50% overlap"
# scheme.
POS_STEP_FRAMES = 1


def pos_signal(r_signal, g_signal, b_signal, fps,
                window_seconds=POS_WINDOW_SECONDS, step_frames=POS_STEP_FRAMES):
    """POS (Wang et al., 2016), sliding-window version matching the original
    paper: instead of normalizing against one average over the whole video,
    slide a short window across the signal, normalize/project each window
    against its OWN local statistics, and overlap-add the results. This
    keeps the signal responsive to lighting/motion that drifts over the
    course of the video, which a single global average would miss."""
    rgb = np.stack([np.array(r_signal), np.array(g_signal), np.array(b_signal)], axis=1)  # frames x 3
    n_frames = rgb.shape[0]
    window_len = max(2, int(round(window_seconds * fps)))
    step_frames = max(1, step_frames)

    h = np.zeros(n_frames)
    for n in range(window_len, n_frames, step_frames):
        m = n - window_len

        window = rgb[m:n, :]  # window_len x 3
        normalized = window / np.mean(window, axis=0)  # each channel vs its OWN mean in this window

        s1 = normalized[:, 1] - normalized[:, 2]                              # Gn - Bn
        s2 = -2 * normalized[:, 0] + normalized[:, 1] + normalized[:, 2]      # -2Rn + Gn + Bn

        std_s2 = np.std(s2)
        alpha = np.std(s1) / std_s2 if std_s2 > 1e-8 else 0
        window_signal = s1 + alpha * s2

        # overlap-add: each frame gets contributions from every window that covered it
        h[m:n] += window_signal - np.mean(window_signal)

    return h


METHODS = {
    "green_baseline": green_baseline_signal,
    "chrom": chrom_signal,
    "pos": pos_signal,
}


def compute_bpm(signal, fps):
    """Turn a pulse-like signal into a BPM estimate using an FFT."""
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        raise ValueError("No face was detected in this video.")
    if len(signal) < fps * 2:
        raise ValueError("Video too short to estimate a heart rate.")

    signal = signal - np.mean(signal)  # remove the constant brightness offset
    signal = signal * np.hamming(len(signal))  # taper edges to reduce FFT artifacts

    fft_magnitudes = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fps)
    bpm_per_freq = freqs * 60

    valid = (bpm_per_freq >= MIN_BPM) & (bpm_per_freq <= MAX_BPM)
    if not np.any(valid):
        raise ValueError("No plausible heart rate found in this video.")

    peak_index = np.argmax(fft_magnitudes[valid])
    bpm = bpm_per_freq[valid][peak_index]
    return round(float(bpm), 1)
