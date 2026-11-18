"""
Case report export for FakeWave: turns ONE cached analysis into a durable,
standalone evidence artifact (a self-contained HTML file plus a JSON sidecar).

Everything here is built from the analysis result the user was looking at (held
in gr.State) -- nothing re-runs analyze_timeline, so the report cannot disagree
with the screen. Images are embedded as base64 data URIs and the HTML has no
external CSS, fonts or scripts, so it renders offline.

Output goes under reports/, which is gitignored: reports carry evidence
filenames, hashes and examiner notes and must never be committed.
"""

import base64
import html
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

import timeline
from preprocess import CONFIG

TOOL_NAME = "FakeWave"
TOOL_VERSION = "0.2.0"

LIMITATIONS = (
    "This report records the output of an automated classifier. It estimates whether "
    "the analyzed audio contains synthetic (AI-generated) speech. It does not identify "
    "the speaker, does not determine which synthesis system produced any segment, and "
    "does not establish the origin, location, or transmission path of the recording. "
    "Reported values are uncalibrated model outputs, not statistical likelihoods. "
    'Segments marked "inconclusive" fall between the configured thresholds and should '
    "be read as neither finding. This output is investigative support and requires "
    "corroboration by independent evidence."
)

_PIPELINE_KEYS = [
    "target_sr", "target_duration_sec", "target_rms_dbfs", "n_fft",
    "hop_length", "n_mels", "mp3_bitrate_kbps", "silence_top_db",
]


def pipeline_params(config=CONFIG):
    """The preprocessing parameters that pin exactly how a result was produced."""
    return {k: config[k] for k in _PIPELINE_KEYS}


def sha256_file(path, chunk_size=1 << 20):
    """Hex SHA-256 of a file, streamed so large files are never fully in memory."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def png_data_uri(rgb_array):
    """(H, W, 3) uint8 array -> 'data:image/png;base64,...'."""
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb_array, dtype=np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_report_data(result, metadata, source_path, model_path,
                      case_ref="", examiner="", notes="", config=CONFIG):
    """JSON-serializable report dict (no image data). `result` is the cached
    analyze_timeline dict, `metadata` the cached clip_metadata dict."""
    now_utc = datetime.now(timezone.utc)
    return {
        "report": {
            "tool": TOOL_NAME,
            "tool_version": TOOL_VERSION,
            "generated_utc": now_utc.isoformat(timespec="seconds"),
            "generated_local": now_utc.astimezone().isoformat(timespec="seconds"),
        },
        "case": {
            "case_reference": case_ref or "",
            "examiner": examiner or "",
            "notes": notes or "",
        },
        "evidence": {
            "filename": Path(source_path).name,
            "sha256": metadata["sha256"],
            "size_bytes": metadata["size_bytes"],
            "format": metadata.get("format"),
            "subtype": metadata.get("subtype"),
            "samplerate": metadata.get("samplerate"),
            "channels": metadata.get("channels"),
            "duration_sec": metadata.get("duration_sec"),
        },
        "analysis": {
            "model_file": Path(model_path).name,
            "model_sha256": sha256_file(model_path),
            "analyzed_duration_sec": result["analyzed_duration_sec"],
            "window_sec": result.get("window_sec", config["target_duration_sec"]),
            "hop_sec": result.get("hop_sec"),
            "n_windows": result["n_windows"],
            "suspect_threshold": timeline.SUSPECT_THRESHOLD,
            "clear_threshold": timeline.CLEAR_THRESHOLD,
            "pipeline": pipeline_params(config),
        },
        "findings": {
            "headline": result["headline"],
            "mean_fake_prob": result["mean_fake_prob"],
            "max_fake_prob": result["max_fake_prob"],
            "n_suspect": result["n_suspect"],
            "n_windows": result["n_windows"],
            "segments": [
                {k: w[k] for k in ("index", "start_sec", "end_sec", "fake_prob", "verdict")}
                for w in result["windows"]
            ],
        },
        "limitations": LIMITATIONS,
    }


_CSS = """
:root { color-scheme: light; }
body { font-family: Arial, Helvetica, sans-serif; color: #1b1b1b; background: #fff;
       max-width: 960px; margin: 24px auto; padding: 0 16px; line-height: 1.45; }
h1 { margin-bottom: 0; }
h2 { border-bottom: 2px solid #ccc; padding-bottom: 4px; margin-top: 32px; }
.sub { color: #555; margin-top: 4px; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ccc; padding: 5px 8px; text-align: left; vertical-align: top; }
th { background: #f0f0f0; }
.kv th { width: 220px; }
.mono { font-family: Consolas, "Courier New", monospace; word-break: break-all; }
.band { padding: 16px 18px; border-radius: 6px; color: #fff; }
.band .headline { font-size: 1.8em; font-weight: 700; }
.pre { white-space: pre-wrap; }
.exhibit img { max-width: 100%; border: 1px solid #ccc; }
.pair { display: flex; gap: 12px; }
.pair figure { flex: 1; margin: 0; }
.pair img { width: 100%; image-rendering: pixelated; }
figcaption, .caption { font-size: 0.9em; color: #444; margin-top: 4px; }
tr.synthetic td { background: #f8d7d7; }
tr.inconclusive td { background: #fdf0c4; }
.limits { border: 1px solid #999; padding: 10px 14px; background: #f7f7f7; }
@media print {
  body { max-width: none; margin: 0; }
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  table, tr, figure, .band, .limits { page-break-inside: avoid; break-inside: avoid; }
  h2 { page-break-after: avoid; break-after: avoid; }
}
"""


def _e(value):
    """Escape anything that came from the user or the filesystem; None -> 'n/a'."""
    return "n/a" if value is None else html.escape(str(value), quote=True)


def _num(v, spec):
    return "n/a" if v is None else format(v, spec)


def _kv_table(rows):
    body = "".join(f"<tr><th>{html.escape(k)}</th><td>{v}</td></tr>" for k, v in rows)
    return f'<table class="kv">{body}</table>'


def render_report_html(data, timeline_png, spectrogram_png, gradcam_png, selected_label):
    """Complete standalone HTML document; images are data URIs. `*_png` arguments
    are data URIs as returned by png_data_uri()."""
    rep, case, ev = data["report"], data["case"], data["evidence"]
    an, fi = data["analysis"], data["findings"]
    color = timeline.HEADLINE_COLORS[fi["headline"]]

    case_tbl = _kv_table([
        ("Case reference", _e(case["case_reference"])),
        ("Examiner", _e(case["examiner"])),
        ("Notes", f'<div class="pre">{_e(case["notes"])}</div>'),
    ])
    ev_tbl = _kv_table([
        ("File name", _e(ev["filename"])),
        ("SHA-256", f'<span class="mono">{_e(ev["sha256"])}</span>'),
        ("Size", f'{ev["size_bytes"]:,} bytes'),
        ("Container format", _e(ev["format"])),
        ("Sample format", _e(ev["subtype"])),
        ("Sample rate", f'{_e(ev["samplerate"])} Hz'),
        ("Channels", _e(ev["channels"])),
        ("Duration", f'{_num(ev["duration_sec"], ".2f")} s'),
    ])
    findings = (
        f'<div class="band" style="background:{color}">'
        f'<div class="headline">{_e(fi["headline"])}</div>'
        f'<div>mean p(synthetic) <b>{fi["mean_fake_prob"]:.3f}</b> &nbsp;|&nbsp; '
        f'max <b>{fi["max_fake_prob"]:.3f}</b> &nbsp;|&nbsp; '
        f'{fi["n_suspect"]} of {fi["n_windows"]} segments flagged '
        f'(p &ge; {an["suspect_threshold"]})</div></div>'
    )
    rows = "".join(
        f'<tr class="{_e(s["verdict"])}"><td>{s["index"]}</td><td>{s["start_sec"]:.2f}</td>'
        f'<td>{s["end_sec"]:.2f}</td><td>{s["fake_prob"]:.4f}</td><td>{_e(s["verdict"])}</td></tr>'
        for s in fi["segments"]
    )
    seg_tbl = (
        "<table><thead><tr><th>#</th><th>start (s)</th><th>end (s)</th>"
        f"<th>p(synthetic)</th><th>verdict</th></tr></thead><tbody>{rows}</tbody></table>"
    )
    pipe = an["pipeline"]
    an_tbl = _kv_table([
        ("Model file", _e(an["model_file"])),
        ("Model SHA-256", f'<span class="mono">{_e(an["model_sha256"])}</span>'),
        ("Analyzed duration (after silence trim)", f'{an["analyzed_duration_sec"]:.2f} s'),
        ("Window / hop", f'{an["window_sec"]} s / {_num(an["hop_sec"], ".3f")} s'),
        ("Segments", _e(an["n_windows"])),
        ("Flag thresholds", f'synthetic &ge; {an["suspect_threshold"]}, authentic &le; {an["clear_threshold"]}'),
        ("Pipeline", "<br>".join(f"{html.escape(k)} = {_e(v)}" for k, v in pipe.items())),
    ])

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FakeWave case report</title>
<style>{_CSS}</style></head><body>
<h1>FakeWave case report</h1>
<p class="sub">Generated {_e(rep["generated_utc"])} (UTC) / {_e(rep["generated_local"])} (local) &middot; {_e(rep["tool"])} {_e(rep["tool_version"])}</p>

<h2>Case details</h2>
{case_tbl}

<h2>Evidence identification</h2>
{ev_tbl}

<h2>Findings</h2>
{findings}

<h2>Timeline exhibit</h2>
<div class="exhibit"><img src="{timeline_png}" alt="Timeline of p(synthetic) per segment">
<div class="caption">p(synthetic) for each {an["window_sec"]}-second window across the analyzed clip. Dashed lines mark the two thresholds.</div></div>

<h2>Selected segment exhibit</h2>
<div class="pair">
<figure><img src="{spectrogram_png}" alt="Mel spectrogram"><figcaption>Mel spectrogram &mdash; {_e(selected_label)}</figcaption></figure>
<figure><img src="{gradcam_png}" alt="Grad-CAM overlay"><figcaption>Grad-CAM overlay &mdash; {_e(selected_label)}</figcaption></figure>
</div>

<h2>Per-segment scores</h2>
{seg_tbl}

<h2>Analysis parameters</h2>
{an_tbl}

<h2>Limitations</h2>
<div class="limits">{_e(data["limitations"])}</div>
</body></html>
"""


def write_report(data, html_str, out_dir="reports"):
    """Write <stem>.html and <stem>.json under out_dir; returns (html_path, json_path)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(data["report"]["generated_utc"]).strftime("%Y%m%dT%H%M%SZ")
    stem = f"fakewave_report_{stamp}_{data['evidence']['sha256'][:8]}"
    html_path, json_path = out / f"{stem}.html", out / f"{stem}.json"
    html_path.write_text(html_str, encoding="utf-8")
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(html_path), str(json_path)
