"""
Sliding-window timeline analysis for FakeWave.

Instead of judging one 4-second window from the middle of a clip, score every
4-second window across the whole (conditioned) clip and report where in time the
suspicious regions are.

Pipeline symmetry: each window goes through the SAME steps the model was trained
on -- whole-file conditioning (preprocess.condition_audio: decode, resample, mp3
round-trip, silence trim), then per-window crop/pad -> RMS-normalize -> log-mel
(preprocess.crop_or_pad / rms_normalize / log_mel, used directly). Nothing here
branches on a label and nothing is written to disk.

This module imports from preprocess and gradcam and modifies neither.
"""

import hashlib
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import soundfile as sf
import torch

import gradcam
from preprocess import CONFIG, condition_audio, crop_or_pad, log_mel, rms_normalize

WINDOW_HOP_SEC = 2.0      # 50% overlap at a 4.0s window
MAX_WINDOWS = 120         # cap cost on long clips by widening the hop, never by truncating
SUSPECT_THRESHOLD = 0.65  # >= this -> flagged synthetic
CLEAR_THRESHOLD = 0.35    # <= this -> flagged authentic; between the two -> inconclusive

VERDICT_COLORS = {"authentic": "#2E7D32", "inconclusive": "#F9A825", "synthetic": "#C62828"}
HEADLINE_COLORS = {
    "Likely authentic": VERDICT_COLORS["authentic"],
    "Inconclusive": VERDICT_COLORS["inconclusive"],
    "Likely synthetic": VERDICT_COLORS["synthetic"],
}


def window_bounds(n_samples, sr, window_sec, hop_sec, max_windows):
    """(start_sample, end_sample) for each analysis window. Never empty.

    Short clips (<= one window) yield exactly one window; crop_or_pad pads it.
    Otherwise windows step by hop_sec, with a final window flush with the end of
    the clip if the regular stride would miss the tail. If that would exceed
    max_windows the hop is widened to (duration - window) / (max_windows - 1), so
    coverage still spans the whole clip -- audio is never dropped to fit the cap.
    """
    window_len = int(round(window_sec * sr))
    if n_samples <= window_len:
        return [(0, n_samples)]

    last_start = n_samples - window_len
    hop = max(1, int(round(hop_sec * sr)))
    starts = list(range(0, last_start + 1, hop))
    if starts[-1] != last_start:
        starts.append(last_start)  # tail window ending exactly at n_samples

    if len(starts) > max_windows:
        if max_windows < 2:
            starts = [0]  # degenerate cap; a single window cannot span the clip
        else:
            # widened hop = (duration - window) / (max_windows - 1)
            wide_hop = last_start / (max_windows - 1)
            starts = [int(round(i * wide_hop)) for i in range(max_windows)]
            starts[-1] = last_start

    return [(s, s + window_len) for s in starts]


def window_logmel(wav, sr, start, end, config=CONFIG):
    """Steps 5-7 for one window: slice, crop/pad, RMS-normalize, log-mel.

    Returns (1, n_mels, time). RMS is computed on the window itself, matching
    training, where it is computed on the 4s crop and not on the whole clip.
    """
    seg = wav[start:end]
    seg = crop_or_pad(seg, sr, config["target_duration_sec"])
    seg = rms_normalize(seg, config["target_rms_dbfs"])
    return log_mel(seg, sr, config["n_fft"], config["hop_length"], config["n_mels"])


def _verdict(p):
    if p >= SUSPECT_THRESHOLD:
        return "synthetic"
    if p <= CLEAR_THRESHOLD:
        return "authentic"
    return "inconclusive"


def _headline(mean_p):
    # Mean, not max: with 100+ windows a single outlier would decide every verdict.
    return {"synthetic": "Likely synthetic", "authentic": "Likely authentic",
            "inconclusive": "Inconclusive"}[_verdict(mean_p)]


def analyze_timeline(path, model, config=CONFIG, hop_sec=WINDOW_HOP_SEC,
                     max_windows=MAX_WINDOWS, progress_cb=None):
    """Score every window of a clip with forward passes only (no Grad-CAM).

    progress_cb, if given, is called as progress_cb(done, total) after each window.
    """
    _, _, wav = condition_audio(path, config)
    sr = config["target_sr"]
    bounds = window_bounds(len(wav), sr, config["target_duration_sec"], hop_sec, max_windows)

    model.eval()
    windows = []
    with torch.no_grad():
        for i, (start, end) in enumerate(bounds):
            batch = window_logmel(wav, sr, start, end, config).unsqueeze(0)  # (1,1,mels,time)
            p = float(torch.sigmoid(model(batch)).item())
            windows.append({
                "index": i, "start_sec": start / sr, "end_sec": end / sr,
                "fake_prob": p, "verdict": _verdict(p),
            })
            if progress_cb is not None:
                progress_cb(i + 1, len(bounds))

    probs = np.array([w["fake_prob"] for w in windows])
    mean_p = float(probs.mean())
    return {
        "conditioned_wav": wav,
        "sr": sr,
        "analyzed_duration_sec": len(wav) / sr,
        "window_sec": config["target_duration_sec"],
        # effective hop (widened on long clips); None when the clip fits in one window
        "hop_sec": (windows[1]["start_sec"] - windows[0]["start_sec"]) if len(windows) > 1 else None,
        "windows": windows,
        "n_windows": len(windows),
        "n_suspect": int((probs >= SUSPECT_THRESHOLD).sum()),
        "mean_fake_prob": mean_p,
        "max_fake_prob": float(probs.max()),
        "headline": _headline(mean_p),
    }


def window_audio(wav, sr, start, end):
    """The conditioned audio for one window, as Gradio wants it: (sr, int16 array).

    This is post-resample, post-codec-normalization, post-silence-trim audio --
    i.e. what the classifier was actually fed, NOT the original recording. It is
    deliberately not RMS-normalized: normalizing per window would flatten the
    loudness differences between segments and make them sound misleadingly alike.

    Window times are relative to the trimmed waveform, so they do not map to
    timestamps in the original file (the leading-silence offset is not recorded).
    """
    seg = np.asarray(wav[start:end], dtype=np.float32)
    if seg.size == 0:
        return sr, np.zeros(1, dtype=np.int16)
    seg = np.clip(seg, -1.0, 1.0)
    return sr, np.round(seg * 32767.0).astype(np.int16)


def explain_window(wav, sr, start, end, model, config=CONFIG):
    """Grad-CAM for one window -> (fake_prob, spectrogram_rgb, overlay_rgb)."""
    logmel = window_logmel(wav, sr, start, end, config)
    batch = logmel.unsqueeze(0)

    cam_engine = gradcam.GradCAM(model)
    try:
        cam, _label, _conf, probs = cam_engine(batch)
    finally:
        cam_engine.remove()

    spectrogram_rgb = gradcam.render_spectrogram_image(logmel)
    overlay_rgb = gradcam.overlay_cam_on_spectrogram(spectrogram_rgb, cam)
    return probs["fake"], spectrogram_rgb, overlay_rgb


def render_timeline_strip(result, width_px=1100, height_px=130):
    """Timeline of per-window p(synthetic) as an (H, W, 3) uint8 RGB array.

    Each window is a bar spanning its own time range with height = fake_prob,
    coloured by verdict. Overlapping windows blend (semi-transparent), and dashed
    lines mark the two decision thresholds.
    """
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    try:
        for w in result["windows"]:
            ax.bar(
                w["start_sec"], w["fake_prob"], width=w["end_sec"] - w["start_sec"],
                align="edge", color=VERDICT_COLORS[w["verdict"]], alpha=0.55,
                edgecolor=VERDICT_COLORS[w["verdict"]], linewidth=0.6,
            )
        for y in (CLEAR_THRESHOLD, SUSPECT_THRESHOLD):
            ax.axhline(y, color="0.35", linestyle="--", linewidth=0.7)

        end = max(w["end_sec"] for w in result["windows"])
        ax.set_xlim(0, end)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel("p(synthetic)", fontsize=8)
        ax.set_xlabel("time (s)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(
            handles=[Patch(color=VERDICT_COLORS["authentic"], alpha=0.7, label="authentic"),
                     Patch(color=VERDICT_COLORS["inconclusive"], alpha=0.7, label="inconclusive"),
                     Patch(color=VERDICT_COLORS["synthetic"], alpha=0.7, label="synthetic")],
            loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=7, frameon=False,
            borderaxespad=0,
        )
        fig.subplots_adjust(left=0.07, right=0.88, top=0.95, bottom=0.30)
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    finally:
        plt.close(fig)


def clip_metadata(path):
    """File hash/size plus whatever soundfile can report. Never raises on
    undecodable containers (m4a/webm fail sf.info but decode fine via ffmpeg)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    meta = {
        "sha256": h.hexdigest(), "size_bytes": os.path.getsize(path),
        "duration_sec": None, "samplerate": None, "channels": None,
        "format": None, "subtype": None,
    }
    try:
        info = sf.info(str(path))
        meta.update(duration_sec=float(info.duration), samplerate=info.samplerate,
                    channels=info.channels, format=info.format, subtype=info.subtype)
    except Exception:
        pass
    return meta
